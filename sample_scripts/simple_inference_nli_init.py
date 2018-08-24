import torch

import src.datasets.data as data
import src.utils.conf as conf
import src.utils.devices as devices
import src.runners.inference as inference
import src.models.nli as nli


def run_inference(model_path, data_path, beam_k=None, output_path=None):
    loaded_model = torch.load(
        model_path
    )

    config = loaded_model["config"]
    model = loaded_model["model"]
    env = conf.EnvConfiguration.from_env()
    device = devices.device_from_env(env)

    word_embeddings, dictionary = data.resolve_embeddings_and_dictionary(
        data_vocab_path=env.data_dict[config.dataset_name]["vocab_path"],
        max_vocab=config.max_vocab,
        vector_cache_path=env.vector_cache_path,
        vector_file_name=config.vector_file_name,
        device=device,
        num_oov=config.num_oov,
        verbose=False,
    )
    word_embeddings.learned_embeddings = model.learned_embeddings
    corpus = data.resolve_corpus(
        data_path=data_path,
        max_sentence_length=200,
    )
    nli_model = nli.get_nli_model(
        nli_code_path=env.nli_code_path,
        nli_pickle_path=env.nli_pickle_path,
        glove_path=env.vector_cache_path / env.nli_vector_file_name,
        word_list=dictionary.word2id.keys(),
        verbose=False,
    )
    inf = inference.Inference(
        model=model, dictionary=dictionary,
        word_embeddings=word_embeddings, device=device,
        config=config, beam_k=beam_k,
    )

    if output_path is None:
        def write(string):
            print(string)
    else:
        f = open(output_path, "w")

        def write(string):
            f.write(string + "\n")

    for translations, all_logprobs, sent_batch in \
            inf.corpus_inference_nli_init(corpus, lambda _: _//2 + 1,
                                          batch_size=16, nli_model=nli_model):
        oov_dicts = dictionary.get_oov_dicts(sent_batch)
        for line in dictionary.ids2sentences(
                translations, oov_dicts, oov_fallback=True):
            write(line)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model-path')
    parser.add_argument('--data-path')
    parser.add_argument('--beam-k', default=None, type=int)
    parser.add_argument('--output-path', default=None)
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        beam_k=args.beam_k,
        output_path=args.output_path,
    )

"""
python sample_scripts/simple_inference.py \
  --model-path {model_path} \
  --data-path {data_path}
"""
