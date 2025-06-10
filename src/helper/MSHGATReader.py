# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils
from utils.load import ConHyperGraphList
from helper.BaseReader import BaseReader


class MSHGATReader(BaseReader):
    def __init__(self, args):
        super().__init__(args)
        
        graph_path = os.path.join(args.path, args.dataset)
        self.diffusion_graph = ConHyperGraphList(graph_path)