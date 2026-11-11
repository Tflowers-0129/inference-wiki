(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.KVDecodeModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const BLOCK_SIZE = 4;
  const POSITION = 8;
  const PHYSICAL_BLOCKS = [7, 2, 9];
  const ALLOCATE_STEP = 1;
  const WRITE_STEP = 2;
  const FIRST_READ_STEP = 3;
  const MAX_STEP = 6;

  function stateAt(step) {
    if (!Number.isInteger(step) || step < 0 || step > MAX_STEP) {
      throw new RangeError('step must be an integer from 0 to ' + MAX_STEP);
    }

    const logicalBlock = Math.floor(POSITION / BLOCK_SIZE);
    const offset = POSITION % BLOCK_SIZE;
    const allocated = step >= ALLOCATE_STEP;
    const blockTable = PHYSICAL_BLOCKS.slice(0, logicalBlock + Number(allocated));
    const writeSlot = allocated ? blockTable[logicalBlock] * BLOCK_SIZE + offset : null;
    const validCount = step >= WRITE_STEP ? POSITION + 1 : POSITION;

    const pages = PHYSICAL_BLOCKS.map((physicalBlock, pageIndex) => ({
      physicalBlock,
      allocated: pageIndex < blockTable.length,
      slots: Array.from({ length: BLOCK_SIZE }, (_, slotOffset) => {
        const position = pageIndex * BLOCK_SIZE + slotOffset;
        return position < validCount ? position : null;
      }),
    }));

    // Reveal reads in logical block order for the animation. Kernels may read in parallel.
    const readPageCount = Math.max(0, step - FIRST_READ_STEP + 1);
    const readPositions = pages.slice(0, readPageCount)
      .flatMap(page => page.slots.filter(position => position !== null));

    return {
      step,
      blockSize: BLOCK_SIZE,
      position: POSITION,
      blockTable,
      writeSlot,
      pages,
      readPositions,
      attentionReady: readPositions.length === POSITION + 1,
      outputReady: step === MAX_STEP,
    };
  }

  return { MAX_STEP, stateAt };
});
