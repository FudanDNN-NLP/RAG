# Retrieval-Augmented Generation Project

## Introduction

This repository is dedicated to the development of the Retrieval-Augmented Generation (RAG) project. The goal of this project is to enhance natural language generation with the ability to retrieve and leverage relevant information from a given dataset or corpus. This approach aims to improve the relevance, accuracy, and richness of the generated text.

## Branching Strategy

To facilitate effective collaboration and streamline the development process, we adopt the following branching strategy:

- **main**: This branch serves as the production branch. It will contain the project's latest stable version. Only finalized and fully tested code should be merged into `main`.

- **develop**: This is our pre-production branch. It acts as an integration branch for features, bug fixes, and other enhancements. All completed features will be merged into `develop` for further testing before merging into `main`.

- **feature/<module_name>**: Developers should create a new branch for each module or feature they are working on. Branch names should follow the format `feature/<module_name>`, where `<module_name>` is a brief identifier for the feature or module (e.g., `feature/evaluation`). This allows for isolated development and testing of individual modules.

## Contribution Guidelines

1. **Create a Branch for Your Module**: If you're starting work on a new module, create a new branch following the `feature/<module_name>` format. For example, if you're working on the LLM's evaluation, your branch could be named `feature/evaluation`.

2. **Commit Your Changes**: Make regular commits to your branch with clear, descriptive commit messages. This not only helps in tracking progress but also assists in understanding the history of changes.

3. **Open a Pull Request (PR)**: Once your module is ready and tested, open a PR against the `develop` branch. Please make sure your PR title clearly describes the feature or fix, and provide a detailed description of the changes in the PR body.
