import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import src.datasets.data as data
import src.models.length_control as length_control
import src.utils.operations as operations


class Inference:
    def __init__(self, model, dictionary, word_embeddings, device, config, beam_k=None):
        self.model = model
        self.dictionary = dictionary
        self.word_embeddings = word_embeddings
        self.device = device
        self.config = config
        self.beam_k = beam_k

    def corpus_inference(self, corpus, desired_length_func, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        for sent_batch in corpus.batch_generator(batch_size):
            desired_length = [
                desired_length_func(len(sentence.split()))
                for sentence in sent_batch
            ]
            translations, all_logprobs = self.batch_inference(
                sent_batch, desired_length)
            yield (translations, all_logprobs, sent_batch)

    def corpus_inference_nli_init(
            self, corpus, desired_length_func, nli_model, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        for sent_batch in corpus.batch_generator(batch_size):
            desired_length = [
                desired_length_func(len(sentence.split()))
                for sentence in sent_batch
            ]
            translations, all_logprobs = self.batch_inference_nli_init(
                sent_batch, desired_length, nli_model
            )
            yield (translations, all_logprobs, sent_batch)

    def single_nli_inference(self, sentence, nli_model,
                             min_desired_length, max_desired_length,
                             device,
                             return_top=None, nli_init=False):
        desired_lengths = np.arange(min_desired_length, max_desired_length)
        batch_size = len(desired_lengths)
        sent_batch = [sentence] * batch_size
        if nli_init:
            translations, log_probs = self.batch_inference_nli_init(
                sent_batch=sent_batch,
                desired_lengths=desired_lengths,
                nli_model=nli_model,
            )
        else:
            translations, log_probs = self.batch_inference(
                sent_batch=sent_batch,
                desired_lengths=desired_lengths,
            )
        pred_samples, pred_lengths, _ = nli_model.encoder.prepare_samples(
            self.dictionary.ids2sentences(translations),
            bsize=self.config.batch_size,
            tokenize=True,
            verbose=False,
        )
        target_samples, target_lengths, _ = nli_model.encoder.prepare_samples(
            sent_batch,
            bsize=self.config.batch_size,
            tokenize=True,
            verbose=False,
        )
        pred_embeds = device(nli_model.encoder.get_batch(pred_samples))
        target_embeds = device(nli_model.encoder.get_batch(target_samples))
        logprobs = nli_model(
            (Variable(pred_embeds), pred_lengths),
            (Variable(target_embeds), target_lengths),
        )
        good_prob = F.softmax(logprobs, dim=1).data.cpu().numpy()[:, 0]
        rankings = np.argsort(good_prob)[::-1]

        if return_top is None:
            return_top = len(rankings)
        ranked_translations = [
            translations[ranked]
            for ranked in rankings[:return_top]
        ]
        scores = [
            good_prob[ranked]
            for ranked in rankings[:return_top]
        ]
        return ranked_translations, scores

    def batch_inference(self, sent_batch, desired_lengths):
        self.model.eval()
        batch_size = len(sent_batch)

        ids, lengths = self.dictionary.sentences2ids(
            sent_batch, sos=False, eos=True)
        ids_batch = self.device(Variable(torch.LongTensor(ids), volatile=True))
        hidden = self.device(self.model.encoder.initial_hidden(batch_size))
        length_input, length_target = \
            length_control.get_fixed_length_data(
                lengths=desired_lengths,
                max_length=self.config.max_length,
                batch_size=batch_size,
            )

        hidden, context = self.model.encoder(
            enc_input=ids_batch,
            lengths=lengths,
            word_embeddings=self.word_embeddings,
            hidden=hidden,
            ids_not_onehots=True,
        )
        if self.beam_k is None:
            all_logprobs, translations = self.model.decoder.forward_decode(
                hidden=hidden,
                context=context,
                input_lengths=lengths,
                length_input=length_input,
                generator=self.model.generator,
                word_embeddings=self.word_embeddings,
                device=self.device,
                config=self.config,
            )
        else:
            all_logprobs, translations = self.model.decoder.forward_decode_beam_search(
                hidden=hidden,
                context=context,
                input_lengths=lengths,
                length_input=length_input,
                generator=self.model.generator,
                word_embeddings=self.word_embeddings,
                device=self.device,
                config=self.config,
                beam_k=self.beam_k,
            )

        return translations, all_logprobs

    def batch_inference_nli_init(self, sent_batch, desired_lengths, nli_model):
        self.model.eval()
        batch_size = len(sent_batch)

        ids, lengths, oov_dicts = self.dictionary.sentences2ids(
            sent_batch, sos=False, eos=True)
        ids_batch = self.device(Variable(torch.LongTensor(ids), volatile=True))
        hidden = self.device(self.model.encoder.initial_hidden(batch_size))
        length_input, length_target = \
            length_control.get_fixed_length_data(
                lengths=desired_lengths,
                max_length=self.config.max_length,
                batch_size=batch_size,
            )

        hidden, context = self.model.encoder(
            enc_input=ids_batch,
            lengths=lengths,
            word_embeddings=self.word_embeddings,
            hidden=hidden,
            ids_not_onehots=True,
        )
        if self.config.nli_mapper_mode in {0, 1, 2}:
            decoder_init = self.model.decoder.nli_mapper(self.device(Variable(
                torch.FloatTensor(nli_model.encoder.encode(sent_batch))
            )))
        elif self.config.nli_mapper_mode == 3:
            decoder_init = self.device(self.model.decoder.initial_output(batch_size))
            hidden = self.model.decoder.nli_mapper(hidden, self.device(Variable(
                torch.FloatTensor(nli_model.encoder.encode(sent_batch))
            )))
        else:
            raise KeyError
        if self.beam_k is None:
            all_logprobs, translations = self.model.decoder.forward_decode(
                hidden=hidden,
                context=context,
                input_lengths=lengths,
                length_input=length_input,
                generator=self.model.generator,
                word_embeddings=self.word_embeddings,
                device=self.device,
                config=self.config,
                decoder_init=decoder_init,
            )
        else:
            all_logprobs, translations = self.model.decoder.forward_decode_beam_search(
                hidden=hidden,
                context=context,
                input_lengths=lengths,
                length_input=length_input,
                generator=self.model.generator,
                word_embeddings=self.word_embeddings,
                device=self.device,
                config=self.config,
                decoder_init=decoder_init,
                beam_k=self.beam_k,
            )

        return translations, all_logprobs

    def corpus_nli_ranked_inference(self,
                                    corpus, nli_model,
                                    min_length_func, max_length_func,
                                    filter_min_length_func=None,
                                    filter_max_length_func=None):
        if filter_min_length_func is None:
            filter_min_length_func = min_length_func
        if filter_max_length_func is None:
            filter_max_length_func = max_length_func

        for i, sent_batch in enumerate(corpus.batch_generator(1)):
            sentence = sent_batch[0]
            length = len(sentence.split())
            ranked_translations, _ = self.single_nli_inference(
                sentence=sentence,
                nli_model=nli_model,
                min_desired_length=min_length_func(length),
                max_desired_length=max_length_func(length),
                device=self.device,
            )
            filter_min_length = filter_min_length_func(length)
            filter_max_length = filter_max_length_func(length)
            yielded = False
            for translation_ids in ranked_translations:
                translation_length = len(translation_ids)
                if filter_min_length <= translation_length <= filter_max_length:
                    yielded = True
                    yield self.dictionary.ids2sentence(translation_ids)
                    break
            if not yielded:
                print(f"No viable summary for sentence {i}: {sentence}")
                yield sentence
