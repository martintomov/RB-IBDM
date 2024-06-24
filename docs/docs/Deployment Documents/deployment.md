---
sidebar_position: 2
---
# Deployment

This page serves as a guide on how to deploy the Docusaurus application in order to call a server that will inference the model for you

**This page covers the basic process workflow.**

# Setup

Prerequisites

1. **Docker Engine/Desktop**: Ensure you are running this notebook in Google Colab for GPU support.

## Deployment

1. Clone the repo and change dir

```bash
git clone https://github.com/martintmv-git/RB-IBDM.git
cd docs
```

2. Build the image and run the container

```bash
docker build -t rb-ibdm-docs .
# Run the Container
docker run -d --name rb-ibdm--app -p 7860:7860 rb-ibdm-docs
```

### Open the UI web app

Open http://localhost:7860/ on your browser.
