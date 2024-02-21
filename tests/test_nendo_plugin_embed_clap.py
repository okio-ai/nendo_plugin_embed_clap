import unittest

import numpy as np
from nendo import Nendo, NendoConfig

nd = Nendo(
    config=NendoConfig(
        log_level="INFO",
        plugins=["nendo_plugin_embed_clap"],
    ),
)


def consine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_arr1 = np.linalg.norm(vec1)
    norm_arr2 = np.linalg.norm(vec2)
    return dot_product / (norm_arr1 * norm_arr2)


class EmbedClapTests(unittest.TestCase):
    def test_funk_closer_than_bark(self):
        nd.library.reset(force=True)
        dog_bark = nd.library.add_track(file_path="tests/assets/barking_dog.wav")
        funny_funk = nd.library.add_track(file_path="tests/assets/funny_funky.mp3")
        hip_hop_funky = nd.library.add_track(file_path="tests/assets/hip_hop_funky.wav")

        dog_bark_embed = nd.plugins.embed_clap(track=dog_bark)
        funny_funk_embed = nd.plugins.embed_clap(track=funny_funk)
        hip_hop_funky_embed = nd.plugins.embed_clap(track=hip_hop_funky)

        self.assertTrue(
            consine_similarity(dog_bark_embed.embedding, funny_funk_embed.embedding)
            < consine_similarity(
                funny_funk_embed.embedding, hip_hop_funky_embed.embedding
            )
        )

    def test_bark_prompt_closer_than_funk(self):
        nd.library.reset(force=True)
        dog_bark = nd.library.add_track(file_path="tests/assets/barking_dog.wav")
        funny_funk = nd.library.add_track(file_path="tests/assets/funny_funky.mp3")

        dog_bark_embed = nd.plugins.embed_clap(track=dog_bark)
        funny_funk_embed = nd.plugins.embed_clap(track=funny_funk)
        _, bark_text_embed = nd.plugins.embed_clap(text="a barking dog")

        self.assertTrue(
            consine_similarity(bark_text_embed, funny_funk_embed.embedding)
            < consine_similarity(dog_bark_embed.embedding, bark_text_embed)
        )

    def test_funk_prompt_closer_than_bark(self):
        nd.library.reset(force=True)
        dog_bark = nd.library.add_track(file_path="tests/assets/barking_dog.wav")
        funny_funk = nd.library.add_track(file_path="tests/assets/funny_funky.mp3")

        dog_bark_embed = nd.plugins.embed_clap(track=dog_bark)
        funny_funk_embed = nd.plugins.embed_clap(track=funny_funk)
        _, funk_text_embed = nd.plugins.embed_clap(text="funky music with guitar")

        self.assertTrue(
            consine_similarity(funk_text_embed, funny_funk_embed.embedding)
            > consine_similarity(dog_bark_embed.embedding, funk_text_embed)
        )


if __name__ == "__main__":
    unittest.main()
