import argparse
import attr
import copy
import inspect
import json
import os
import sys
import pathlib


def _is_true(x):
    return x == "True"


def argparse_attr(default=attr.NOTHING, validator=None,
                  repr=True, cmp=True, hash=True, init=True,
                  convert=None, opt_string=None,
                  **argparse_kwargs):
    if opt_string is None:
        opt_string_ls = []
    elif isinstance(opt_string, str):
        opt_string_ls = [opt_string]
    else:
        opt_string_ls = opt_string

    if argparse_kwargs.get("type", None) is bool:
        argparse_kwargs["choices"] = {True, False}
        argparse_kwargs["type"] = _is_true

    return attr.attr(
        default=default,
        validator=validator,
        repr=repr,
        cmp=cmp,
        hash=hash,
        init=init,
        convert=convert,
        metadata={
            "opt_string_ls": opt_string_ls,
            "argparse_kwargs": argparse_kwargs,
        }
    )


def update_parser(parser, class_with_attributes):
    for attribute in class_with_attributes.__attrs_attrs__:
        if "argparse_kwargs" in attribute.metadata:
            argparse_kwargs = attribute.metadata["argparse_kwargs"]
            opt_string_ls = attribute.metadata["opt_string_ls"]
            if attribute.default is attr.NOTHING:
                argparse_kwargs = argparse_kwargs.copy()
                argparse_kwargs["required"] = True
            else:
                argparse_kwargs["default"] = attribute.default
            parser.add_argument(
                f"--{attribute.name}", *opt_string_ls,
                **argparse_kwargs
            )


def read_parser(parser, class_with_attributes, skip_non_class_attributes=False):
    attribute_name_set = {
        attribute.name
        for attribute in class_with_attributes.__attrs_attrs__
    }

    kwargs = dict()
    leftover_kwargs = dict()

    for k, v in vars(parser.parse_args()).items():
        if k in attribute_name_set:
            kwargs[k] = v
        else:
            if not skip_non_class_attributes:
                raise RuntimeError(f"Unknown attribute {k}")
            leftover_kwargs[k] = v

    instance = class_with_attributes(**kwargs)
    if skip_non_class_attributes:
        return instance, leftover_kwargs
    else:
        return instance


@attr.s
class EnvConfiguration:
    sys_path = attr.attr()
    vector_cache_path = attr.attr()
    data_dict = attr.attr()
    log_folder_path = attr.attr()
    model_save_path = attr.attr()
    nli_code_path = attr.attr()
    nli_pickle_path = attr.attr()
    nli_vector_file_name = attr.attr()
    use_gpu = attr.attr()

    @classmethod
    def from_json(cls, json_string):
        return cls(**json.loads(json_string))

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path, "r") as f:
            return cls.from_json(f.read())

    def __attrs_post_init__(self):
        self.vector_cache_path = pathlib.Path(self.vector_cache_path)
        self.data_dict = {
            dataset_name: {
                "data_path": path_or_none(dataset_dict.get("data_path")),
                "vocab_path": path_or_none(dataset_dict.get("vocab_path")),
                "top_words_path": path_or_none(
                    dataset_dict.get("top_words_path")),
            }
            for dataset_name, dataset_dict in self.data_dict.items()
        }
        self.log_folder_path = pathlib.Path(self.log_folder_path)
        self.model_save_path = pathlib.Path(self.model_save_path)

    @classmethod
    def from_env(cls):
        try:
            env_config_path = os.environ["NLU_ENV_CONFIG_PATH"]
        except KeyError:
            raise RuntimeError("Set up your NLU_ENV_CONFIG_PATH")
        return cls.from_json_path(env_config_path)


@attr.s
class Configuration:
    # Operations
    run_name = argparse_attr(
        default="ANON_RUN", type=str)
    dataset_name = argparse_attr(
        default="billion", type=str)
    env_config_path = argparse_attr(
        default="", type=str)
    vector_file_name = argparse_attr(
        default="glove.840B.300d.txt", type=str)
    max_vocab = argparse_attr(
        default=20000, type=int)
    max_sentence_length = argparse_attr(
        default=100, type=int)
    num_oov = argparse_attr(
        default=10, type=int)
    log_step = argparse_attr(
        default=50, type=int)
    model_save_every = argparse_attr(
        default=5000, type=int)

    # Training
    learning_rate = argparse_attr(
        default=0.0003, type=float)
    learning_rate_mult = argparse_attr(
        default=1., type=float)
    learning_rate_mult_every = argparse_attr(
        default=10000000000, type=float)
    batch_size = argparse_attr(
        default=32, type=int)
    max_steps = argparse_attr(
        default=10000000, type=int)
    autoencode_perc = argparse_attr(
        default=0.2, type=float)
    gradient_clipping = argparse_attr(
        default=5.0, type=float)
    length_countdown = argparse_attr(
        default="normal", type=str)

    # Loss multipliers
    length_penalty_multiplier = argparse_attr(
        default=1., type=float)
    autoencode_loss_multiplier = argparse_attr(
        default=1., type=float)
    nli_loss_multiplier = argparse_attr(
        default=0., type=float)

    # Model: General
    hidden_size = argparse_attr(
        default=512, type=int)
    encoder_birectional = argparse_attr(
        default=True, type=bool)
    encoder_nlayers = argparse_attr(
        default=3, type=int)
    decoder_nlayers = argparse_attr(
        default=3, type=int)
    decoder_type = argparse_attr(
        default="attn", type=str, choices={"attn", "rnn"})
    rnn_type = argparse_attr(
        default="gru", type=str, choices={"gru", "lstm"})
    encoder_dropout = argparse_attr(
        default=0.0, type=float)
    decoder_dropout = argparse_attr(
        default=0.0, type=float)
    decoder_generator = argparse_attr(
        default="embedding", type=str, choices={"linear", "embedding"})
    min_length = argparse_attr(
        default=2, type=int)
    max_length = argparse_attr(
        default=100, type=int)
    max_ratio = argparse_attr(
        default=1.5, type=float)
    init_decoder_with_nli = argparse_attr(
        default=False, type=bool)
    nli_mapper_mode = argparse_attr(
        default=0, type=int)

    # Model: DAE
    ae_noising = argparse_attr(
        default=True, type=bool)
    ae_noising_kind = argparse_attr(
        default=None, type=str)
    ae_add_noise_perc_per_sent_low = argparse_attr(
        default=0.25, type=float)
    ae_add_noise_perc_per_sent_high = argparse_attr(
        default=0.45, type=float)
    ae_add_noise_num_sent = argparse_attr(
        default=2, type=int)
    ae_add_noise_2_grams = argparse_attr(
        default=False, type=bool)

    # --- Placeholders
    env = None

    def __attrs_post_init__(self):
        if self.env_config_path:
            self.env = EnvConfiguration.from_json_path(self.env_config_path)
        else:
            self.env = EnvConfiguration.from_env()

        self.add_sys_path()

    @classmethod
    def parse_configuration(cls, prog=None, description=None):
        parser = argparse.ArgumentParser(
            prog=prog,
            description=description,
        )
        update_parser(
            parser=parser,
            class_with_attributes=cls,
        )
        return read_parser(
            parser=parser,
            class_with_attributes=cls,
        )

    def to_dict(self):
        config_dict = {}
        for attribute in inspect.getfullargspec(Configuration)[0]:
            if attribute == "self":
                continue
            config_dict[attribute] = getattr(self, attribute)
        return config_dict

    def to_json(self):
        serialized_dict = self.to_dict()
        for key, val in serialized_dict.items():
            if isinstance(val, pathlib.Path):
                serialized_dict[key] = str(val)
            if isinstance(val, EnvConfiguration):
                pass
        return json.dumps(serialized_dict, indent=2)

    @classmethod
    def from_json(cls, json_string):
        return cls(**json.loads(json_string))

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path, "r") as f:
            return cls.from_json(f.read())

    @classmethod
    def from_json_arg(cls):
        assert len(sys.argv) == 2
        return cls.from_json_path(sys.argv[1])

    def copy(self):
        return copy.deepcopy(self)

    def add_sys_path(self):
        if self.env.sys_path:
            sys.path += [self.env.sys_path]


def path_or_none(path):
    if path is not None:
        return pathlib.Path(path)
    else:
        return path
