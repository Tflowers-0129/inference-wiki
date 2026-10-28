/* Educational model: one sequence per request; decode first; FIFO chunked prefill.
   Prefill completion produces output token #1. Each later decode round produces one.
   Rounds are discrete scheduling events, not equal wall-clock intervals. */
(function (root) {
  'use strict';
  function simulate({output = 1024, total = 200, maxSeqs = 200, budget = 8192, input = 4096} = {}) {
    if (![output,total,maxSeqs,budget,input].every(Number.isInteger) || Math.min(output,total,maxSeqs,budget,input) < 1) throw new Error('Model parameters must be positive integers.');
    const requests = Array.from({length: total}, (_, i) => ({id: i + 1, inputLeft: input, outputLeft: output - 1, started: false, done: false, inputComputed: 0, decoded: 0}));
    const steps = [];
    let complete = 0;
    while (complete < total) {
      if (steps.length > 20000) throw new Error('Simulation exceeded its safety limit.');
      const decoding = requests.filter(r => r.started && !r.done && r.inputLeft === 0);
      if (decoding.length > budget) throw new Error('This teaching model requires a token budget at least as large as its decode batch.');
      let remaining = budget - decoding.length;
      let slots = maxSeqs - decoding.length;
      let finishedNow = 0;
      const chunks = [];
      for (const r of decoding) {
        r.outputLeft--; r.decoded++;
        if (!r.outputLeft) {r.done = true; complete++; finishedNow++;}
      }
      for (const r of requests) {
        if (!remaining || !slots) break;
        if (r.done || !r.inputLeft) continue;
        const before = r.inputLeft;
        const tokens = Math.min(before, remaining);
        const continued = r.started;
        r.started = true; r.inputLeft -= tokens; r.inputComputed += tokens;
        remaining -= tokens; slots--;
        chunks.push({id:r.id,tokens,continued,left:r.inputLeft,complete:r.inputLeft === 0});
        if (!r.inputLeft && output === 1) {r.done = true; complete++; finishedNow++;}
      }
      const active = requests.filter(r => r.started && !r.done).length;
      const waiting = requests.filter(r => !r.started).length;
      const prefilling = requests.filter(r => r.started && !r.done && r.inputLeft > 0).length;
      steps.push({number:steps.length + 1,decode:decoding.length,chunks,prefill:chunks.reduce((sum,c)=>sum+c.tokens,0),used:budget - remaining,free:remaining,active,waiting,prefilling,complete,finishedNow,admitted:total-waiting,scheduled:decoding.length+chunks.length});
    }
    const peak = Math.max(...steps.map(s=>s.active));
    return {steps,requests,peak,peakStep:steps.find(s=>s.active===peak).number,output,total,budget,input,maxSeqs};
  }
  root.WikiScheduler = {simulate};
})(globalThis);
