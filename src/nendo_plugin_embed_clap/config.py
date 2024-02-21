"""Configuration for the EmbedClap plugin."""
from nendo import NendoConfig
from pydantic import Field


class EmbedClapConfig(NendoConfig):
    """Configuration for the EmbedClap plugin.

    Attributes:
        model (str): The model to use for embedding. Defaults to "laion/larger_clap_general".
        max_embed_length (int): The maximum length of the embedding. Defaults to 512.
        sample_rate (int): The sample rate to use for the embedding. Defaults to 44100.

    """

    model: str = Field("laion/larger_clap_general")
    max_embed_length: int = 512
    sample_rate: int = 44100
