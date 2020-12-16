# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict
from sklearn import metrics

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank, evaluate_py_clothes
from .rerank import re_ranking
from .roc import evaluate_roc
from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir
        self.use_clothes = cfg.TEST.USE_ONLY_CLO

        self.features = []
        self.pids = []
        self.camids = []
        if self.use_clothes:
            self.cloids = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []
        if self.use_clothes:
            self.cloids = []
        

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        if self.use_clothes:
            self.cloids.extend(inputs['clothids'])
        self.features.append(outputs.cpu())

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            if self.use_clothes:
                cloids = comm.gather(self.cloids)
                cloids = sum(cloids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            # num_query + num_gallery
            features = self.features 
            pids = self.pids
            camids = self.camids
            if self.use_clothes:
                cloids = self.cloids

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        if self.use_clothes:
            query_cloids = np.asarray(cloids[:self._num_query])
            gallery_cloids = np.asarray(cloids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        # Cosine default
        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        if self.use_clothes:
            cmc, all_AP, all_INP = evaluate_py_clothes(dist, query_pids, gallery_pids, query_camids, gallery_camids, query_cloids, gallery_cloids)
        else:
            cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)
