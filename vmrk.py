#!/usr/bin/python

"""
Process one VMRK file into a set of summary statistics.

TODO optionally filter outliers
TODO allow selected blocks to be dropped

The processing is done in two steps.  First run process_vmrk to obtain
a condensed form of the VMRK data, then run summarize_vmrk to obtain
the summary statistics.
"""

import csv
import numpy as np
import os
import sys
import logging
from collections import OrderedDict

# Outliers are defined to fall outside the interval (low, high)
low = 150
high = 3000

# Should be 1
ddof = 0

class Code(object):
    """
    A stimulus/response code
    """
    def __init__(self, side, congruent, correct):
        self.side = side
        self.congruent = congruent
        self.correct = correct

    def __str__(self):
        return "side=%s congruent=%r correct=%r" % (
            self.side, self.congruent, self.correct)

    def fromSRCodes(a, b):
        if a in (1, 2):
            side = "left"
        else:
            side = "right"
        if a in (1, 3):
            congruent = True
        else:
            congruent = False
        if b in (5, 6):
            correct = True
        else:
            correct = False

        return Code(side, congruent, correct)


class Trial(object):
    """
    Raw information about one trial.
    """
    def __init__(self, srcode, time):
        self.srcode = srcode
        self.time = time

    def __str__(self):
        return "srcode=%s time=%d" % (self.srcode, self.time)


class Block(object):
    """
    Holds all information about a block of trials
    """
    def __init__(self):
        self.Code = []    # Stimulus/response codes
        self.Rtim = []    # Response times
        self.Ntri = []    # Number of responses within a trial

    def filter_outliers(self, low, high):
        ii = [i for i, rt in enumerate(self.Rtim) if (rt >= low) and (rt <= high)]
        self.Code = [self.Code[i] for i in ii]
        self.Rtim = [self.Rtim[i] for i in ii]
        self.Ntri = [self.Ntri[i] for i in ii]

    def query(self, side=None, congruent=None, correct=None, lastcorrect=None):
        """
        Return all response times with a given stimulus/response code pair.
        """
        ret = []
        for j, (c, x) in enumerate(zip(self.Code, self.Rtim)):
            if side is not None and c.side != side:
                continue
            if congruent is not None and c.congruent != congruent:
                continue
            if correct is not None and c.correct != correct:
                continue
            if lastcorrect is not None and j > 0:
                if self.Code[j-1].correct != lastcorrect:
                    continue
            ret.append(x)

        return ret

    def postCorrectRtim(self):
        """
        Return all response times that follow a correct response.
        """
        ret = []
        for k in range(1, len(self.Rtim)):
            if self.KY[k-1][1] in (5, 6):
                ret.append(self.Rtim[k])
        return ret

    def postErrorRtim(self):
        """
        Return all response times that follow an error response.
        """
        ret = []
        for k in range(1, len(self.Rtim)):
            if self.KY[k-1][1] in (5, 6):
                ret.append(self.Rtim[k])
        return ret

    def __str__(self):
        print(self.RT)

def process_trial(qu, block):
    """
    Insert summary values obtained from qu into block.

    Parameters
    ----------
    qu :
        Data from a single trial.
    block :
        All data grouped by trial/response type
    """

    # A trial must start with S99 (fixation cross), then have a
    # stimulus and response.  Return early if this is not the case.
    if len(qu) < 3 or qu[0].srcode != 99:
        return

    # ky is the stimulus and response type
    code = Code.fromSRCodes(qu[1].srcode, qu[2].srcode)
    logging.info("Debug Info: %s" % code)

    # Response time for first response, multiplication by 2 is a scale
    # conversion.
    rt = 2 * (qu[2].time - qu[1].time)
    block.Code.append(code)
    block.Rtim.append(rt)
    logging.info("Debug Info: %s" % qu[1].time)
    block.Ntri.append(len(qu))

    block.filter_outliers(low, high)


def collapse_blocks(data):
    """
    Collapse a list of blocks into a single block.
    """
    blk = Block()
    for block in data:
        blk.Code.extend(block.Code)
        blk.Rtim.extend(block.Rtim)
        blk.Ntri.extend(block.Ntri)
    return blk


def summarize_vmrk(filename, data):
    """
    Create summary statistics from a VMRK file.

    The VMRK file must be processed with process_vmrk before running
    this function.
    """

    cdata = collapse_blocks(data)

    results = OrderedDict()

    results["sid"] = filename.split(".")[0]

    all_trials = cdata.Rtim
    correct_trials = [x for k,x in zip(cdata.Code, cdata.Rtim) if k.correct]
    error_trials = [x for k,x in zip(cdata.Code, cdata.Rtim) if not k.correct]

    results["fcn"] = len(correct_trials)
    results["fen"] = len(error_trials)
    results["facc"] = 100 * results["fcn"] / len(all_trials)

    # All trial summaries
    results["frtm"] = np.mean(all_trials)
    results["frtsd"] = np.std(all_trials, ddof=ddof)

    # All correct trial summaries
    results["frtmc"] = np.mean(correct_trials)
    results["frtsdc"] = np.std(correct_trials, ddof=ddof)

    # All error trial summaries
    results["frtme"] = np.mean(error_trials)
    results["frtsde"] = np.std(error_trials, ddof=ddof)

    # Congruent correct trials
    v = cdata.query(correct=True, congruent=True)
    results["fccn"] = len(v)
    results["fcrtmc"] = np.mean(v)
    results["fcrtsdc"] = np.std(v, ddof=ddof)

    # Congruent error trials
    v = cdata.query(correct=False, congruent=True)
    results["fcen"] = len(v)
    results["fcrtme"] = np.mean(v)
    results["fcrtsde"] = np.std(v, ddof=ddof)

    # Congruent accuracy
    results["fcacc"] = 100 * results["fccn"] / (results["fccn"] + results["fcen"])

    # Incongruent correct trials
    v = cdata.query(correct=True, congruent=False)
    results["ficn"] = len(v)
    results["firtmc"] = np.mean(v)
    results["firtsdc"] = np.std(v, ddof=ddof)

    # Incongruent error trials
    v = cdata.query(correct=False, congruent=False)
    results["fien"] = len(v)
    results["firtme"] = np.mean(v)
    results["firtsde"] = np.std(v, ddof=ddof)

    # Incongruent accuracy
    results["fiacc"] = 100 * results["ficn"] / (results["ficn"] + results["fien"])

    # Post correct correct trials
    # (don't count first trial of each block)
    v = [b.query(correct=True, lastcorrect=True) for b in data]
    results["fpccn"] = sum([max(0, len(x) - 1) for x in v])

    # Post correct error trials
    # (don't count first trial of each block)
    v = [b.query(correct=False, lastcorrect=True) for b in data]
    results["fpcen"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpcen"] > 0:
        results["fpcertm"] = sum([sum(x[1:]) for x in v]) / results["fpcen"]
    else:
        results["fpcertm"] = 0.

    # Post error correct trials
    # (don't count first trial of each block)
    v = [b.query(correct=True, lastcorrect=False) for b in data]
    results["fpecn"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpecn"] > 0:
        results["fpecrtm"] = sum([sum(x[1:]) for x in v]) / results["fpecn"]
    else:
        results["fpecrtm"] = 0.

    # Post error error trials
    # (don't count first trial of each block)
    v = [b.query(correct=False, lastcorrect=False) for b in data]
    results["fpeen"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpeen"] > 0:
        results["fpeertm"] = sum([sum(x[1:]) for x in v]) / results["fpeen"]
    else:
        results["fpeertm"] = 0.

    # Post error any trials
    # (don't count first trial of each block)
    v = [b.query(lastcorrect=False) for b in data]
    results["fpexn"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpexn"] > 0:
        results["fpexrtm"] = sum([sum(x[1:]) for x in v]) / results["fpexn"]
    else:
        results["fpexrtm"] = 0.

    # Post any error trials
    # (don't count first trial of each block)
    v = [b.query(correct=False) for b in data]
    results["fpxen"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpxen"] > 0:
        results["fpxertm"] = sum([sum(x[1:]) for x in v]) / results["fpxen"]
    else:
        results["fpxertm"] = 0.

    # Post correct accuracy
    results["faccpc"] = results["fpccn"] / (results["fpccn"] + results["fpcen"])

    # Post error accuracy
    results["faccpe"] = results["fpecn"] / (results["fpecn"] + results["fpeen"])

    # Post error slowing
    results["fpes"] = results["fpeertm"] - results["fpecrtm"]
    results["fpes2"] = results["fpcertm"] - results["fpecrtm"]
    results["fpes3"] = results["fpxertm"] - results["fpexrtm"]

    # Anticipatory responses
    results["fan"] = np.sum(np.asarray(all_trials) < 150)
    results["faen"] = np.sum(np.asarray(error_trials) < 150)

    # Trials with extra responses
    results["fscn"] = sum(np.asarray(cdata.Ntri) > 3)

    return results


def process_vmrk(filename):
    """
    Process the VMRK format file with name filename.

    Parameters
    ----------
    filename :
        Name of a vmrk format file.

    Returns
    -------
    data : list of Blocks
        data[j] contains all the data for block j.
    """

    fid = open(filename)
    rdr = csv.reader(fid)

    # Keep track of which block we are in
    blocknum = 0
    dblock = 0

    # Assume that we start in practice mode
    mode = "practice"
    n99 = 0
    n144 = False
    qu, data = [], []
    block = Block()

    for line in rdr:

        # Only process "mark" lines
        if len(line) == 0 or not line[0].startswith("Mk"):
            logging.info("Skipping row: %s" % line)
            continue

        # Lines have format Mk###=type, where type=comment, stimulus
        f0 = line[0].split("=")
        fl = f0[1].lower()
        if fl == "comment":
            continue
        elif fl == "stimulus":
            pass
        else:
            # Not sure what else exists, log it and move on
            logging.info("Skipping row: %s" % line)
            continue

        # Get the type code, e.g. if S16 then n=16
        f1 = line[1].replace(" ", "")
        stimcode = int(f1[1:])

        if mode == "practice":
            if stimcode == 99:
                n99 += 1
            if n99 == 3 and stimcode == 144:
                n144 = True
            if n99 == 3 and n144 and stimcode == 255:
                mode = "experiment"
                continue

        if mode == "practice":
            continue

        qu.append(Trial(stimcode, int(line[2])))

        # Handle end of block markers
        if stimcode in (144, 255):
            if dblock > 0:
                process_trial(qu[0:-1], block)
                qu = [qu[-1]]
                blocknum += 1
                dblock = 0
                data.append(block)
                block = Block()
            continue
        dblock += 1

        if stimcode == 99:
            process_trial(qu[0:-1], block)
            qu = [qu[-1]]

    # Final trial may not have been processed
    process_trial(qu, block)

    return data


if __name__ == "__main__":

    import csv

    logging.basicConfig(filename="vmrk.log", level=logging.DEBUG)

    results = []
    # Run all .vmrk files found in the current directory
    vmrk_files = []
    for file in os.listdir('.'):
        if file.find('.vmrk') > 0:
            vmrk_files.append(file)

    for i, fname in enumerate(vmrk_files):
        data = process_vmrk(fname)
        result = summarize_vmrk(fname, data)

        if i == 0:
            wtr = csv.writer(sys.stdout, lineterminator='\n')
            header = [k for k in result]
            wtr.writerow(header)

        wtr.writerow([result[k] for k in result])

