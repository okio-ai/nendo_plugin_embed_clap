# Nendo Plugin Embed CLAP

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="nendo core">
</p>
<br>

<p align="left">
<a href="https://okio.ai" target="_blank">
    <img src="https://img.shields.io/website/https/okio.ai" alt="Website">
</a>
<a href="https://twitter.com/okio_ai" target="_blank">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai" alt="Twitter">
</a>
<a href="https://discord.gg/gaZMZKzScj" target="_blank">
    <img src="https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat" alt="Discord">
</a>
</p>

---

A plugin to create joint embeddings from text and audio using CLAP by LAION.

## Features 

- Create joint embeddings from text and audio
- Use the embeddings in combination with nendo's vector search functionality

## Requirements

Please make sure you have the correct version of Pytorch installed. 
Go to [pytorch.org](https://pytorch.org/get-started/locally/) and select your OS, 
package manager and CUDA version to get the correct installation command.


## Installation

1. [Install Nendo](https://github.com/okio-ai/nendo#installation)
2. `pip install nendo-plugin-embed-clap`

## Usage

Take a look at a basic usage example below.
For more detailed information, please refer to the [documentation](https://okio.ai/docs/plugins).

```pycon
>>> from nendo import Nendo
>>> nd = Nendo(plugins=["nendo_plugin_embed_clap"])
>>> track = nd.library.add_track(file_path="path/to/file.mp3")

>>> embedding = nd.plugins.embed_clap(track=track)
```

## Contributing

Visit our docs to learn all about how to contribute to Nendo: [Contributing](https://okio.ai/docs/contributing/)


## License

Nendo: MIT License

Pretrained models: The weights are released by LAION under the Apache 2.0 license.
