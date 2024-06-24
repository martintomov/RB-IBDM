---
sidebar_position: 2
---
# Deployment

This page serves as a guide on how to deploy the Gradio application in order to call a server that will inference the model for you

**This page covers the basic process workflow.**

# Setup

Prerequisites

1. **Docker Engine/Desktop**: Ensure you are running this notebook in Google Colab for GPU support.

## Deployment

1. Clone the repo and change dir

```bash
git clone https://github.com/martintmv-git/RB-IBDM.git
cd gradio-apps/gradio-detectron
```

2. Build the image and run the container

```bash
docker build -t detectron-server .
# Run the Container
docker run -d --name insectsam-app -p 7860:7860 detectron-server
```

### Open the UI web app

Open http://localhost:7860/ on your browser.
