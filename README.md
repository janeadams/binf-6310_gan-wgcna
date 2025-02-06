# BINF 6310: GAN-WGCNA
Group Members: Jane Adams, Madison Nguyen, Nicole Leon Vargas, Tzu-Tung (Bella) Chang

## Project Summary

> [!NOTE]  
> To do: Add paper and data source links

The purpose of this project is to reproduce a published work in bioinformatics. we selected "GAN-WGCNA: Calculating gene modules to identify key intermediate regulators in cocaine addiction." This study stood out for its innovative integration of Weighted Gene Co-expression Network Analysis (WGCNA) and Generative Adversarial Networks (GANs)—two well-established but distinct computational techniques. The combination of these methods offers a powerful approach to uncovering gene expression patterns linked to phenotypic traits, making it an exciting case study in applying machine learning to biological data. The study’s focus on cocaine addiction was particularly compelling, as it demonstrates how computational methods can enhance our understanding of complex disorders with genetic components. A key factor in our decision was the study’s transparency and accessibility. The authors provided both the dataset and analysis scripts, which are well-structured and documented. This ensures a clearer path for our reproducibility study and allows us to critically examine and validate their findings without unnecessary roadblocks related to missing or proprietary data.

## Project Structure

> [!NOTE]  
> To do: Describe file structure of repo


### Dependencies
> [!WARNING]  
> We are using uv. DO NOT use `pip` directly or create a requirements.txt file. For example, instead of `pip install tqdm`, you should use `uv add tqdm`, or prepend all pip commands, e.g. `uv pip install tqdm`.

This project uses `uv` for dependency management. [Read more about `uv` here.](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) To set up your environment on first clone, run `uv sync`. Learn more about uv project structure [here](https://docs.astral.sh/uv/guides/projects/#project-structure).

To activate the virtual environment after it is created, you can call `source .venv/bin/activate` on Mac/Linux or `.venv\Scripts\activate` on Windows. You might want to just add an alias to your bash config, like `vim .zshrc` > `alias activate="source .venv/bin/activate"` so you can just type `activate` to activate.

## Contributing

> [!NOTE]  
> To do: Explain branching, PRs, etc.
