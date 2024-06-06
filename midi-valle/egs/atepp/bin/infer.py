#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --text-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import argparse
import logging
import os
from pathlib import Path
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater, get_midi_token_collater
from valle.models import get_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-prompts",
        type=str,
        default="",
        help="Text prompts which are separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="To get up and running quickly just follow the steps below.",
        help="Text to be synthesized.",
    )

    # model
    # add_model_arguments(parser)
    # parser.add_argument(
    #     "--text-tokens",
    #     type=str,
    #     default="data/tokenized/unique_text_tokens.k2symbols",
    #     help="Path to the unique text tokens file.",
    # )

    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    parser.add_argument(
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )
    
    parser.add_argument(
        "--demo",
        type=str2bool,
        default=False,
        help="Do demo task.",
    )

    return parser.parse_args()


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    text_tokens = args.text_tokens

    return model, text_tokens


@torch.no_grad()
def main():
    args = get_args()
    # text_tokenizer = TextTokenizer(backend=args.text_extractor)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    model, text_tokens = load_model(args.checkpoint, device)
    midi_collater = get_midi_token_collater()

    audio_tokenizer = AudioTokenizer()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], 24000
                )

            audio_prompts.append(encoded_frames[0][0])

        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)

    text_prompts = np.load(text_prompts)
    
    for n, text in enumerate(args.text.split("|")):
        if text != "":
            text = np.load(text) 
            text_tokens, text_tokens_lens = midi_collater(
                [
                    np.concatenate([text_prompts[:-1], text[1:]], axis=0).tolist()
                ]
            )
        else:
            text_tokens, text_tokens_lens = midi_collater(
                [
                    text_prompts.tolist()
                ]
            )
            
        logging.info(f"synthesize text: {text_tokens}")

        # synthesis
        if args.continual:
            assert text == ""
            encoded_frames = model.continual(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
            )
        else:
            enroll_x_lens = None
            if len(text_prompts) != 0:
                _, enroll_x_lens = midi_collater(
                    [
                        text_prompts[:-1].tolist()
                    ]
                )
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=args.top_k,
                temperature=args.temperature,
            )

        if audio_prompts != []:
            samples = audio_tokenizer.decode(
                [(encoded_frames.transpose(2, 1), None)]
            )
            # store
            torchaudio.save(
                f"{args.output_dir}/{n}.wav", samples[0].cpu(), 24000
            )
        else:  # Transformer
            pass


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
