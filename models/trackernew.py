# coding: utf-8
import torch
from scipy.optimize import linear_sum_assignment
from util import box_ops
import copy
from scipy.spatial.distance import cdist
import numpy as np
class Tracker(object):
    def __init__(self, score_thresh, max_age=300, alpha=0.3, T1=1.2, T2=20):
        self.score_thresh = score_thresh
        self.max_age = max_age        
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
        self.reset_all()
        self.num=13
        self.sum=0

        self.alpha = alpha
        self.T1 = T1
        self.T2 = T2

        self.outlier_history = {}

    def reset_all(self):
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
    
    def init_track(self, results):

        scores = results["scores"]
        classes = results["labels"]
        bboxes = results["boxes"]
        ret = list()
        ret_dict = dict()
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj["tracking_id"] = self.id_count
                obj['active'] = 1
                obj['age'] = 1
                obj['lof'] = []
                obj['count'] = []
                ret.append(obj)
                ret_dict[idx] = obj
        
        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)

    def _kdist(self, arr):
        inds_sort = np.argsort(arr)
        neighbor_ind = inds_sort[1:4]
        return neighbor_ind, arr[neighbor_ind[-1]]

    def get_lrd(self, nei_inds, rdist, NN):
        lrd = np.zeros(NN)
        for i, inds in enumerate(nei_inds):
            s = 0
            for j in inds:
                s += rdist[j, i]
            lrd[i] = 3/s
        return lrd

    def is_mother_pig(self, det_box):
        areas = (det_box[:, 2] - det_box[:, 0]) * (det_box[:, 3] - det_box[:, 1])
        mother_pig_index = torch.argmax(areas).item()
        return mother_pig_index


    def compute_distance_to_mother_boundary(piglets_positions, mother_boundary):
        x_min, y_min, x_max, y_max = mother_boundary
        distances = []

        for x, y in piglets_positions:

            delta_x = max(0, max(x_min - x, x - x_max))
            delta_y = max(0, max(y_min - y, y - y_max))

            dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
            distances.append(dist)

        return torch.tensor(distances)

    def update_outliers(self, outliers, T2):
        if not hasattr(self, 'outlier_state'):
            self.outlier_state = {}

        new_outlier_state = {}
        for pig_id in outliers:
            if pig_id in self.outlier_state:
                new_outlier_state[pig_id] = self.outlier_state[pig_id] + 1
            elif pig_id not in self.outlier_state:
                new_outlier_state[pig_id] = 1
        self.outlier_state = new_outlier_state

    def is_close_to_others(self, outlier_pig_id, dist_to_mother_boundary, dist, avg_length, corresponding):
        outlier_pig_id = corresponding.index(outlier_pig_id)
        if dist_to_mother_boundary[outlier_pig_id].numpy() > avg_length.numpy() and np.sum(
                dist[outlier_pig_id, :] > avg_length.numpy()) == len(dist[outlier_pig_id, :]) - 2:
            return True
        else:
            return False

    def update_outlier_history(self, frame_id, outlier_ids, mother_pig_id):
        outlier_list = [0] * (self.num + 1)
        for outlier_id in outlier_ids:
            outlier_list[outlier_id] = 1
        outlier_list[mother_pig_id] = 0
        self.outlier_history[frame_id] = {'outlier_list': outlier_list, 'mother_pig_id': mother_pig_id}

    def step(self, output_results, left, right, count, T1, T2):
        self.T1 = T1
        self.T2 = T2
        scores = output_results["scores"]
        classes = output_results["labels"]
        bboxes = output_results["boxes"]
        track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None
        
        results = list()
        results1 = list()
        results_dict = dict()

        tracks = list()
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0)
        for idx in range(scores.shape[0]):
            if idx in self.tracks_dict and track_bboxes is not None:
                self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                x_c = (obj['bbox'][0] + obj['bbox'][2]) / 2
                y_c = (obj['bbox'][1] + obj['bbox'][3]) / 2
                obj["center"] = [x_c, y_c]
                obj["lof"]=[]
                obj['count'] = []
                results.append(obj)
                results_dict[idx] = obj

        for idx in range(scores.shape[0]):
            if scores[idx] < self.score_thresh and scores[idx] > 0.1:
                if bboxes[idx, 2] > right or bboxes[idx, 2] < left:
                    obj = dict()
                    obj["score"] = float(scores[idx])
                    obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                    det_box = torch.stack([torch.tensor(obj['bbox'])], dim=0)

                    cost_bbox = 1.0 - box_ops.box_iou1(det_box, track_box)

                    matched_indices = linear_sum_assignment(cost_bbox)
                    for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                        if cost_bbox[m0, m1] < 0.8:

                            det_old = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0)
                            cost_bbox1 = 1.0 - box_ops.box_iou1(det_box, det_old)
                            matched_indices1 = linear_sum_assignment(cost_bbox1)
                            for (m2, m3) in zip(matched_indices1[0], matched_indices1[1]):
                                if cost_bbox1[m2, m3] > 0.9:
                                    x_c = (obj['bbox'][0] + obj['bbox'][2]) / 2
                                    y_c = (obj['bbox'][1] + obj['bbox'][3]) / 2
                                    obj["center"] = [x_c, y_c]
                                    obj["lof"] = []
                                    obj['count'] = []
                                    results.append(obj)
                                    results_dict[idx] = obj
                                    results1.append(obj)
                            if len(results1) > 0:
                                det_old1 = torch.stack([torch.tensor(obj['bbox']) for obj in results1], dim=0)
                                cost_bbox2 = 1.0 - box_ops.generalized_box_iou(det_box, det_old1)
                                matched_indices2 = linear_sum_assignment(cost_bbox2)
                                for (m4, m5) in zip(matched_indices2[0], matched_indices2[1]):
                                    if cost_bbox1[m2, m3] > 0.7 and cost_bbox2[m4, m5] > 0.7 and cost_bbox2[
                                        m4, m5] < 1 and (idx not in results_dict.keys()):
                                        # qiu zhi xin
                                        x_c = (obj['bbox'][0] + obj['bbox'][2]) / 2
                                        y_c = (obj['bbox'][1] + obj['bbox'][3]) / 2
                                        obj["center"] = [x_c, y_c]
                                        obj["lof"] = []
                                        obj['count'] = []
                                        results.append(obj)
                                        results_dict[idx] = obj

        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        N = len(results)
        M = len(tracks)

        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            det_box = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4
            data = torch.stack([torch.tensor(obj['center']) for obj in results], dim=0)

            mother_pig_index = self.is_mother_pig(det_box)
            mother_pig = data[mother_pig_index]
            mother_pig_bbox = det_box[mother_pig_index]  # 母猪的边框
            piglets = np.delete(data, mother_pig_index, axis=0)
            det_box_piglets = np.delete(det_box, mother_pig_index, axis=0)
            avg_length = torch.max(det_box[:, 2] - det_box[:, 0], det_box[:, 3] - det_box[:, 1]).mean()

            dist_to_mother_boundary = Tracker.compute_distance_to_mother_boundary(piglets, mother_pig_bbox) # new sow_dist!!!!

            dist = cdist(piglets.cpu().numpy(), piglets.cpu().numpy())



            dist_to_mother_boundary_2d = np.tile(dist_to_mother_boundary, (dist.shape[1], 1)).T
            weighted_dist = self.alpha * dist + (1 - self.alpha) * dist_to_mother_boundary_2d

            nei_kdist = np.apply_along_axis(self._kdist, 1, weighted_dist)
            nei_inds, kdist = zip(*nei_kdist)
            for i, k in enumerate(kdist):
                ind = np.where(weighted_dist[i] < k)
                weighted_dist[i][ind] = k

            NN = piglets.shape[0]

            lrd = self.get_lrd(nei_inds, weighted_dist, NN)

            score0 = np.zeros(NN)
            for i, inds in enumerate(nei_inds):
                NNN = len(inds)
                lrd_nei = sum(lrd[inds])
                score0[i] = lrd_nei / 3 / lrd[i]
            score0 = np.insert(score0, mother_pig_index, 0)

            out_ind = np.where(score0 > self.T1)[0]
            outliers = data[out_ind]


            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0)
            cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box)

            matched_indices = linear_sum_assignment(cost_bbox)

            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

            matches = [[],[]]
            out_id = []

            for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                if cost_bbox[m0, m1] > 1.2:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)
                else:
                    matches[0].append(m0)
                    matches[1].append(m1)

            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                if 'tracking_id' in tracks[m1]:
                    track['tracking_id'] = tracks[m1]['tracking_id']
                    if m0 in out_ind:
                        out_id.append(tracks[m1]['tracking_id'])
                    track['age'] = 1
                    track['active'] = 1
                    tracks[m1]['lof'].append(score0[m0])
                    tracks[m1]['count'].append(count-1)
                    track['lof']=tracks[m1]['lof']
                    track['count'] = tracks[m1]['count']
                    pre_box = tracks[m1]['bbox']
                    cur_box = track['bbox']
                    ret.append(track)
                else:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)

        for i in unmatched_dets:
            if self.id_count <self.num:
                track = results[i]
                self.id_count += 1
                track['tracking_id'] = self.id_count
                track['age'] = 1
                track['active'] = 1
                track['lof'] = []
                ret.append(track)
            else:
                m=[]

                track_old=list(results_dict.items())[i]
                track_old=track_old[1]
                det_box = torch.stack([torch.tensor(track_old['bbox'])], dim=0)
                for obj in unmatched_tracks:   #
                    m.append(tracks[obj])
                if len(m)>0:
                    track_box = torch.stack([torch.tensor(obj['bbox']) for obj in m], dim=0)  # M x 4
                    cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box)  # N x M

                    matched_indices = linear_sum_assignment(cost_bbox)
                    matches1 = [[], []]
                    for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                        matches1[0].append(m0)
                        matches1[1].append(m1)

                    for (m0, m1) in zip(matches1[0], matches1[1]):
                        track = track_old
                        if 'tracking_id' in m[m1]:
                            track['tracking_id'] = m[m1]['tracking_id']
                            if m0 in out_ind:
                                out_id.append(tracks[m1]['tracking_id'])
                            track['age'] = 1
                            track['active'] = 1
                            track["lof"].append(score0[m0])
                            track["count"].append(count-1)
                            pre_box = tracks[m1]['bbox']
                            cur_box = track['bbox']
                            ret.append(track)
                            del unmatched_tracks[m1]
                            del unmatched_dets[m0]
                        else:
                            mm=[]
                            for obj in unmatched_tracks:
                                if 'tracking_id' in tracks[obj].keys():
                                    mm.append(tracks[obj])
                            if len(mm) > 0:
                                track_box = torch.stack([torch.tensor(obj['bbox']) for obj in mm], dim=0)
                                cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box)

                                matched_indices = linear_sum_assignment(cost_bbox)
                                matches1 = [[], []]
                                for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                                    if cost_bbox[m0, m1] < 1.2:
                                        matches1[0].append(m0)
                                        matches1[1].append(m1)

                                for (m0, m1) in zip(matches1[0], matches1[1]):
                                    track = track_old
                                    track['tracking_id'] = mm[m1]['tracking_id']
                                    if m0 in out_ind:
                                        out_id.append(mm[m1]['tracking_id'])
                                    track['age'] = 1
                                    track['active'] = 1
                                    track["lof"].append(score0[m0])
                                    track["count"].append(count-1)
                                    pre_box = tracks[m1]['bbox']
                                    cur_box = track['bbox']
                                    ret.append(track)
                                    del unmatched_tracks[m1]
                                    del unmatched_dets[m0]

        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            ret_unmatched_tracks.append(track)

        out_ind_result = list(map(lambda x: ret[x]['tracking_id'], out_ind))
        self.update_outliers(out_ind_result, T2)


        out_ind_result = []
        outlier_pig_ids = [pig_id for pig_id, count in self.outlier_state.items() if
                           count >= T2]

        extra = torch.zeros(1)
        tensor1 = dist_to_mother_boundary[:mother_pig_index]
        tensor2 = dist_to_mother_boundary[mother_pig_index:]
        dist_to_mother_boundary1 = torch.cat((tensor1, extra, tensor2), dim=-1)


        n1 = np.insert(dist, mother_pig_index, 0, axis=0)
        n1 = np.insert(n1, mother_pig_index, 0, axis=1)

        corresponding = list(map(lambda x: ret[x]['tracking_id'], range(len(ret))))

        for outlier_pig_id in outlier_pig_ids:
            if self.is_close_to_others(outlier_pig_id, dist_to_mother_boundary1, n1, avg_length, corresponding):
                out_ind_result.append(outlier_pig_id)
        self.update_outlier_history(count, out_ind_result, ret[mother_pig_index]['tracking_id'])

        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        his_track=ret+self.unmatched_tracks
        return copy.deepcopy(ret), outliers, score0, data, out_id, his_track, out_ind_result

