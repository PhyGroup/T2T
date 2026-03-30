#!/usr/bin/env python

import os
import subprocess
import threading
import unicodedata

METEOR_JAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meteor-1.5.jar')

class Meteor:
    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx4G', METEOR_JAR, '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1, universal_newlines=True, encoding='utf-8', errors='ignore'
        )
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert sorted(gts.keys()) == sorted(res.keys())
        imgIds = sorted(gts.keys())
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert len(res[i]) == 1
            hypothesis = self._sanitize_string(res[i][0].strip())
            references = [self._sanitize_string(ref.strip()) for ref in gts[i] if ref.strip()]

            if not hypothesis or len(hypothesis.split()) < 2:
                scores.append(0.0)
                continue
            if not references:
                scores.append(0.0)
                continue

            try:
                stat = self._stat(hypothesis, references)
                if not stat or not stat.strip():
                    scores.append(0.0)
                    continue
                eval_line += ' ||| {}'.format(stat)
            except Exception:
                scores.append(0.0)
                continue

        try:
            self.meteor_p.stdin.write(f"{eval_line}\n")
            self.meteor_p.stdin.flush()
            for i in range(len(imgIds)):
                if len(scores) > i and scores[i] == 0.0:
                    continue
                score_line = self.meteor_p.stdout.readline().strip()
                try:
                    score = float(score_line.split()[-1])
                    if 0 <= score <= 1:
                        scores.append(score)
                    else:
                        scores.append(0.0)
                except (ValueError, IndexError):
                    scores.append(0.0)
            final_line = self.meteor_p.stdout.readline().strip()
            try:
                final_score = float(final_line.split()[-1])
            except (ValueError, IndexError):
                final_score = sum(scores) / len(scores) if scores else 0.0
        except Exception:
            while len(scores) < len(imgIds):
                scores.append(0.0)
            final_score = sum(scores) / len(scores) if scores else 0.0

        self.lock.release()
        return final_score, scores

    def method(self):
        return "METEOR"

    def _sanitize_string(self, s):
        s = s.replace('\ufeff', '').replace('|||', '').replace('\n', ' ').replace('\r', ' ')
        s = unicodedata.normalize('NFKD', s)
        return ' '.join(s.split())

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        try:
            self.meteor_p.stdin.write(f"{score_line}\n")
            self.meteor_p.stdin.flush()
            stat = self.meteor_p.stdout.readline().strip()
            return stat
        except Exception:
            return ""

    def __del__(self):
        try:
            self.lock.acquire()
            self.meteor_p.stdin.close()
            self.meteor_p.stdout.close()
            self.meteor_p.stderr.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
            self.lock.release()
        except Exception:
            pass