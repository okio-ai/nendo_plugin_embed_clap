"""Nendo plugin for creating text and audio embeddings using CLAP."""
from typing import Any, Tuple

import numpy.typing as npt
import torch
from nendo import Nendo, NendoEmbeddingPlugin, NendoConfig, NendoTrack
from transformers import ClapModel, AutoTokenizer, AutoFeatureExtractor

from .config import EmbedClapConfig

settings = EmbedClapConfig()


class EmbedClap(NendoEmbeddingPlugin):
    """A nendo plugin for embedding audio and text with CLAP.

    Can be used in combination with `nendo` vector extensions to find similar audio and text.

    Examples:
        ```python
        from nendo import Nendo, NendoConfig

        nendo = Nendo(config=NendoConfig(plugins=["nendo_plugin_embed_clap"]))
        track = nendo.library.add_track(file_path="path/to/file.wav")

        embedding = nendo.plugins.embed_clap(track=track)
        ```
    """

    nendo_instance: Nendo = None
    config: NendoConfig = None
    tokenizer: Any = None
    feature_extractor: Any = None
    model: ClapModel = None
    device: str = None

    def __init__(self, **data: Any):
        """Initialize the plugin."""
        super().__init__(**data)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ClapModel.from_pretrained(settings.model).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            settings.model,
            sampling_rate=settings.sample_rate,
        )

    @NendoEmbeddingPlugin.run_text
    def embed_text(self, text: str) -> Tuple[str, npt.NDArray]:
        """Embed a text string.

        Args:
            text (str): Text to embed.

        Returns:
            Tuple[str, npt.NDArray]: Tuple of the text and the embedding.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                return_tensors="pt",
                max_length=settings.max_embed_length,
                truncation=True,
            ).to(self.device)
            return text, self.model.get_text_features(**inputs).squeeze(0).cpu().numpy()

    def _track_to_text(self, track: NendoTrack) -> str:
        """Get the text to embed the track with.

        Either use the summary or the track_to_text function to get the text to embed the track with.

        Args:
            track (NendoTrack): Track to embed.

        Returns:
            str: Text to embed the track with.
        """
        summary = track.get_plugin_value("summary")
        if summary is not None:
            return summary

        return self.track_to_text(track)

    @NendoEmbeddingPlugin.run_track
    def embed_track(
        self,
        track: NendoTrack,
        audio_only: bool = False,
    ) -> Tuple[str, npt.NDArray]:
        """Embed a track with its plugin_data as text.

        Either use the summary or the track_to_text function to get the text to embed the track with.
        We then average the audio and text features to get the final embedding.

        Args:
            track (NendoTrack): Track to embed.
            audio_only (bool): Whether to only use audio features.

        Returns:
            Tuple[str, npt.NDArray]: Tuple of the text and the embedding.
        """
        with torch.no_grad():
            if settings.sample_rate != track.sr:
                track.resample(settings.sample_rate)

            signal = track.signal if track.signal.ndim == 1 else track.signal[0]
            audio_inputs = self.feature_extractor(
                signal,
                sampling_rate=settings.sample_rate,
                return_tensors="pt",
            ).to(self.device)
            audio_feats = self.model.get_audio_features(**audio_inputs)

            if audio_only:
                return (
                    "",
                    audio_feats.squeeze(0).cpu().numpy(),
                )

            # TODO use more sophisticated text function
            embedding_text = self._track_to_text(track)
            text_inputs = self.tokenizer(
                embedding_text,
                padding=True,
                return_tensors="pt",
                max_length=settings.max_embed_length,
                truncation=True,
            ).to(self.device)
            text_feats = self.model.get_text_features(**text_inputs)

            # TODO check if we want to do something other than averaging
            return (
                embedding_text,
                torch.mean(torch.cat([audio_feats, text_feats]), dim=0).cpu().numpy(),
            )
