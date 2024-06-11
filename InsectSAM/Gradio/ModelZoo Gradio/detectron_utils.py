import torch
import numpy as np
import cv2
from huggingface_hub import hf_hub_download

REPO_ID = "idml/Detectron2-FasterRCNN_InsectDetect"
FILENAME = "model.pth"
FILENAME_CONFIG = "config.yml"


# Ensure you have the model file

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt


viz_classes = {'thing_classes': ['Acrididae',
                                 'Agapeta',
                                 'Agapeta hamana',
                                 'Animalia',
                                 'Anisopodidae',
                                 'Aphididae',
                                 'Apidae',
                                 'Arachnida',
                                 'Araneae',
                                 'Arctiidae',
                                 'Auchenorrhyncha indet.',
                                 'Baetidae',
                                 'Cabera',
                                 'Caenidae',
                                 'Carabidae',
                                 'Cecidomyiidae',
                                 'Ceratopogonidae',
                                 'Cercopidae',
                                 'Chironomidae',
                                 'Chrysomelidae',
                                 'Chrysopidae',
                                 'Chrysoteuchia culmella',
                                 'Cicadellidae',
                                 'Coccinellidae',
                                 'Coleophoridae',
                                 'Coleoptera',
                                 'Collembola',
                                 'Corixidae',
                                 'Crambidae',
                                 'Culicidae',
                                 'Curculionidae',
                                 'Dermaptera',
                                 'Diptera',
                                 'Eilema',
                                 'Empididae',
                                 'Ephemeroptera',
                                 'Erebidae',
                                 'Fanniidae',
                                 'Formicidae',
                                 'Gastropoda',
                                 'Gelechiidae',
                                 'Geometridae',
                                 'Hemiptera',
                                 'Hydroptilidae',
                                 'Hymenoptera',
                                 'Ichneumonidae',
                                 'Idaea',
                                 'Insecta',
                                 'Lepidoptera',
                                 'Leptoceridae',
                                 'Limoniidae',
                                 'Lomaspilis marginata',
                                 'Miridae',
                                 'Mycetophilidae',
                                 'Nepticulidae',
                                 'Neuroptera',
                                 'Noctuidae',
                                 'Notodontidae',
                                 'Object',
                                 'Opiliones',
                                 'Orthoptera',
                                 'Panorpa germanica',
                                 'Panorpa vulgaris',
                                 'Parasitica indet.',
                                 'Plutellidae',
                                 'Psocodea',
                                 'Psychodidae',
                                 'Pterophoridae',
                                 'Pyralidae',
                                 'Pyrausta',
                                 'Sepsidae',
                                 'Spilosoma',
                                 'Staphylinidae',
                                 'Stratiomyidae',
                                 'Syrphidae',
                                 'Tettigoniidae',
                                 'Tipulidae',
                                 'Tomoceridae',
                                 'Tortricidae',
                                 'Trichoptera',
                                 'Triodia sylvina',
                                 'Yponomeuta',
                                 'Yponomeutidae']}



def detectron_process_image(image):
    cfg = get_cfg()


    cfg.merge_from_file(hf_hub_download(repo_id=REPO_ID, filename=FILENAME_CONFIG))
    cfg.MODEL.WEIGHTS = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.DEVICE='cpu'
    predictor = DefaultPredictor(cfg)

    numpy_image = np.array(image)


    im = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    v = Visualizer(im[:, :, ::-1],
                   viz_classes,
                   scale=0.5)
    outputs = predictor(im)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    results = out.get_image()[:, :, ::-1]
    rgb_image = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)

    return rgb_image
