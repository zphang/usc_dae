import datetime as dt
import numpy as np
import os

import torch

import src.datasets.data as data
import src.models.encoders as encoders
import src.models.decoders as decoders
import src.models.generators as generators
import src.models.top as top
import src.utils.conf as conf
import src.utils.devices as devices
import src.utils.operations as operations
import src.utils.logs as logs
import src.utils.misc as misc
import src.models.length_control as length_control
import src.models.nli as nli
import src.models.autoencode as autoencode


class DAETrainer:
    def __init__(self,
                 config, device,
                 corpus, word_embeddings, dictionary,
                 model, optimizers,
                 nli_model, special_vocab,
                 verbose=False,
                 ):
        assert isinstance(config, conf.Configuration)
        self.config = config
        self.device = device

        self.corpus = corpus
        self.word_embeddings = word_embeddings
        self.dictionary = dictionary

        self.model = model
        self.optimizers = optimizers

        self.nli_model = nli_model
        self.special_vocab = special_vocab

        # state
        self.step = 0
        self.curr_learning_rate = self.config.learning_rate
        self.loss_ls = []
        self.creation_date = dt.datetime.now()
        self.log_path = logs.get_named_date_log_path(
            log_folder_path=self.config.env.log_folder_path,
            run_name=self.config.run_name,
            timestamp=self.creation_date,
        )
        self.logger = logs.SimpleFileLogger(open(self.log_path, "a"))
        if verbose:
            print(self.config.to_json())

        assert self.config.autoencode_perc == 1.0

    def print_tail(self):
        print(f"tail -f {self.log_path}")

    def run_train(self):
        self.print_tail()

        # === 3. Run-loop === #
        self.logger.write_line(self.config.to_json())
        while self.step < self.config.max_steps:

            self.optimizers.zero_grad()
            self.model.train()

            self.autoencode_step()

            self.update_learning_rate()
            self.maybe_save_model()
            self.step += 1

    def update_learning_rate(self):
        new_learning_rate = (
            self.config.learning_rate
            * self.config.learning_rate_mult
            ** (self.step // self.config.learning_rate_mult_every)
        )
        if new_learning_rate != self.curr_learning_rate:
            for optimizer in self.optimizers.optimizer_ls:
                operations.update_learning_rate(
                    optimizer=optimizer,
                    new_learning_rate=new_learning_rate,
                )
            self.logger.write_line(
                f"Updated LR from {self.curr_learning_rate} "
                f"to {new_learning_rate}"
            )
            self.curr_learning_rate = new_learning_rate

    def maybe_save_model(self):
        if self.step > 0 and self.step % self.config.model_save_every == 0:
            self.save_model()

    def save_model(self):
        save_path = (
            self.config.env.model_save_path
            / f"{self.config.run_name}_"
              f"{misc.datetime_format(self.creation_date)}_"
              f"{self.step:06d}.p"
        )
        torch.save({
            "config": self.config,
            "optimizer": self.optimizers,
            "model": self.model,
        }, save_path)
        print(f"Saved to: {save_path}")

    def autoencode_step(self):
        sent_batch = self.corpus.next_batch(self.config.batch_size)

        raw_autoencode_loss, autoencode_logprobs, oov_dicts, input_ids, target_ids, target_lengths = \
            autoencode.get_autoencode_loss(
                sent_batch=sent_batch,
                encoder=self.model.encoder,
                decoder=self.model.decoder,
                generator=self.model.generator,
                corpus=self.corpus,
                dictionary=self.dictionary,
                device=self.device,
                word_embeddings=self.word_embeddings,
                max_length=self.config.max_length,
                ae_noising=self.config.ae_noising,
                ae_noising_kind=self.config.ae_noising_kind,
                init_decoder_with_nli=self.config.init_decoder_with_nli,
                nli_model=self.nli_model,
                nli_mapper_mode=self.config.nli_mapper_mode,
                ae_add_noise_perc_per_sent_low=self.config.ae_add_noise_perc_per_sent_low,
                ae_add_noise_perc_per_sent_high=self.config.ae_add_noise_perc_per_sent_high,
                ae_add_noise_num_sent=self.config.ae_add_noise_num_sent,
                length_countdown_mode=self.config.length_countdown,
            )
        autoencode_loss = (
            raw_autoencode_loss * self.config.autoencode_loss_multiplier
        )

        if self.config.length_penalty_multiplier != 0:
            raw_length_penalty = length_control.get_length_penalty(
                log_probs=autoencode_logprobs,
                length_target=target_lengths,
                output_length=autoencode_logprobs.shape[0],
                device=self.device,
            )
            length_penalty = \
                raw_length_penalty * self.config.length_penalty_multiplier
        else:
            length_penalty = self.device(operations.zero_loss())

        loss = autoencode_loss + length_penalty
        loss.backward()
        operations.clip_gradients(
            model=self.model,
            gradient_clipping=self.config.gradient_clipping,
        )

        self.optimizers.step()

        if self.step % self.config.log_step == 0:
            print(self.step)

            self.logger.write_line(
                f"[{self.step}] "
                f"AUT={autoencode_loss.data[0]:.5f}, LP={length_penalty.data[0]:.5f}, Total={loss.data[0]:.5f}"
            )
            i = np.random.randint(len(target_ids))
            original_sent = self.dictionary.rawids2sentence(
                target_ids[i], oov_dicts[i])
            input_sent = self.dictionary.rawids2sentence(
                input_ids[i], oov_dicts[i])
            output_sent = self.dictionary.rawids2sentence(
                autoencode_logprobs[:, i].max(1)[
                    1].data.cpu().numpy(),
                oov_dicts[i],
            )

            self.logger.write_line(f"ORIG: {original_sent}")
            if self.config.ae_noising:
                self.logger.write_line("---")
                self.logger.write_line(f"NOIS: {input_sent}")
            self.logger.write_line("---")
            self.logger.write_line(f"AUTO: {output_sent}")
            self.logger.write_line("LENGTH: {}/{}/{} = {:.4f}".format(
                data.raw_length(original_sent),
                data.raw_length(input_sent),
                data.raw_length(output_sent),
                data.raw_length(output_sent) / data.raw_length(original_sent),
            ))
            self.logger.write("\n===\n")
            self.logger.flush()

    @classmethod
    def from_config(cls, config):
        device = devices.device_from_conf(config)
        word_embeddings, dictionary = data.resolve_embeddings_and_dictionary(
            data_vocab_path=config.env.data_dict[
                config.dataset_name]["vocab_path"],
            max_vocab=config.max_vocab,
            vector_cache_path=config.env.vector_cache_path,
            vector_file_name=config.vector_file_name,
            device=device,
            num_oov=config.num_oov,
        )
        corpus = data.resolve_corpus(
            data_path=config.env.data_dict[
                config.dataset_name]["data_path"],
            max_sentence_length=config.max_sentence_length,
        )
        special_vocab = autoencode.resolve_special_vocab(
            data_top_words_path=config.env.data_dict[
                config.dataset_name]["top_words_path"],
        )
        encoder = device(encoders.RNNEncoder(
            embedding_size=word_embeddings.embedding_dim,
            bidirectional=config.encoder_birectional,
            hidden_size=config.hidden_size,
            layers=config.encoder_nlayers,
            rnn_type=config.rnn_type,
            dropout=config.encoder_dropout,
        ))
        if config.decoder_type == "attn":
            decoder_used = decoders.AttnRNNDecoder
        elif config.decoder_type == "rnn":
            decoder_used = decoders.RNNDecoder
        else:
            raise KeyError()
        decoder = device(decoder_used(
            embedding_size=word_embeddings.embedding_dim,
            hidden_size=config.hidden_size,
            layers=config.decoder_nlayers,
            rnn_type=config.rnn_type,
            dropout=config.decoder_dropout,
        ))
        generator = generators.resolve_generator(
            decoder_generator=config.decoder_generator,
            hidden_size=config.hidden_size,
            dictionary=dictionary,
            word_embeddings=word_embeddings,
            device=device,
        )
        nli_model = nli.resolve_nli_model(
            nli_code_path=config.env.nli_code_path,
            nli_pickle_path=config.env.nli_pickle_path,
            glove_path=config.env.vector_cache_path / config.env.nli_vector_file_name,
            word_list=dictionary.word2id.keys(),
            nli_loss_multiplier=config.nli_loss_multiplier,
            init_decoder_with_nli=config.init_decoder_with_nli,
            device=device,
        )
        decoder.nli_mapper = device(nli.resolve_nli_mapper(
            init_decoder_with_nli=config.init_decoder_with_nli,
            nli_model=nli_model,
            hidden_size=config.hidden_size,
            nli_mapper_mode=config.nli_mapper_mode,
            rnn_type=config.rnn_type,
        ))
        model = top.SimpleModel(
            encoder=encoder,
            decoder=decoder,
            generator=generator,
            learned_embeddings=word_embeddings.learned_embeddings,
        )
        model.initialize()
        optimizers = model.initialize_optimizer(
            learning_rate=config.learning_rate,
        )
        return cls(
            config=config, device=device,
            corpus=corpus, word_embeddings=word_embeddings,
            dictionary=dictionary,
            model=model, optimizers=optimizers,
            nli_model=nli_model, special_vocab=special_vocab
        )
