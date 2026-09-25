// Streaming player: parses length-prefixed frame packets from the server and
// plays them back on the source timeline with an adaptive jitter buffer, so
// playback stays smooth (never freezes or jumps) even when processing runs
// slower than real time.

const PREBUFFER_S = 0.4;
const TARGET_BUFFER_S = 0.8;

export class StreamPlayer {
  constructor({ onFrame, onEnd, onPacket }) {
    this.onFrame = onFrame;
    this.onEnd = onEnd || (() => {});
    this.onPacket = onPacket || (() => {});
    this.mode = "smooth";
    this.reset();
  }

  reset() {
    if (this.abort) this.abort.abort();
    this.abort = null;
    this.queue = [];
    this.ended = false;
    this.started = false;
    this.playing = true;
    this.clock = 0;
    this.rate = 1;
    this.lastTick = null;
    this.current = null;
    this.decoding = false;
    this.fps = 30;
    this.received = 0;
    this.maxBufferS = 6;
    this._finished = false;
  }

  get bufferedSeconds() {
    if (!this.queue.length) return 0;
    return Math.max(0, this.queue[this.queue.length - 1].header.t - this.clock);
  }

  async open(url) {
    this.reset();
    const controller = new AbortController();
    this.abort = controller;
    let response;
    try {
      response = await fetch(url, { signal: controller.signal, cache: "no-store" });
    } catch (err) {
      if (controller.signal.aborted) return;
      throw err;
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Stream failed (${response.status})`);
    }
    const reader = response.body.getReader();
    let buf = new Uint8Array(0);
    try {
      for (;;) {
        // Back-pressure: stop reading while we hold several seconds of frames.
        while (this.queue.length / this.fps > this.maxBufferS && !controller.signal.aborted) {
          await new Promise((r) => setTimeout(r, 30));
        }
        const { value, done } = await reader.read();
        if (done) break;
        buf = concat(buf, value);
        let offset = 0;
        while (buf.length - offset >= 4) {
          const view = new DataView(buf.buffer, buf.byteOffset + offset);
          const total = view.getUint32(0);
          if (buf.length - offset - 4 < total) break;
          this.push(buf.subarray(offset + 4, offset + 4 + total));
          offset += 4 + total;
        }
        buf = buf.slice(offset);
      }
    } catch (err) {
      if (!controller.signal.aborted) console.warn("stream error", err);
    }
    if (this.abort === controller) this.ended = true;
  }

  push(packet) {
    const view = new DataView(packet.buffer, packet.byteOffset);
    const headLen = view.getUint32(0);
    const header = JSON.parse(new TextDecoder().decode(packet.subarray(4, 4 + headLen)));
    let pos = 4 + headLen;
    const blobs = header.jpeg_sizes.map((size) => {
      const blob = new Blob([packet.slice(pos, pos + size)], { type: "image/jpeg" });
      pos += size;
      return blob;
    });
    this.fps = header.fps || this.fps;
    this.received++;
    this.queue.push({ header, blobs });
    this.onPacket(header);
  }

  tick(now) {
    const dt = this.lastTick == null ? 0 : Math.min((now - this.lastTick) / 1000, 0.1);
    this.lastTick = now;
    if (!this.queue.length) {
      if (this.ended && this.started) this.finish();
      return;
    }
    let pick = null;
    if (this.mode === "edge") {
      pick = this.queue[this.queue.length - 1];
      this.queue = [pick];
      this.clock = pick.header.t;
      this.rate = 1;
      this.started = true;
    } else {
      if (!this.started) {
        const span = this.queue[this.queue.length - 1].header.t - this.queue[0].header.t;
        if (span < PREBUFFER_S && !this.ended) {
          // Show the first processed frame immediately while the buffer fills.
          if (!this.current) this.show(this.queue[0]);
          return;
        }
        this.started = true;
        this.clock = this.queue[0].header.t;
      }
      if (this.playing) {
        const buffered = this.bufferedSeconds;
        if (this.ended) this.rate = Math.min(1, this.rate * 1.05 + 0.01);
        else if (buffered < 0.15) this.rate = Math.max(0.05, this.rate * 0.9);
        else if (buffered < TARGET_BUFFER_S) this.rate = Math.max(0.05, this.rate * 0.995);
        else if (buffered > TARGET_BUFFER_S * 1.5) this.rate = Math.min(1, this.rate * 1.02 + 0.002);
        this.clock += dt * this.rate;
      }
      while (this.queue.length > 1 && this.queue[1].header.t <= this.clock) this.queue.shift();
      pick = this.queue[0];
      if (pick.header.t > this.clock + 1 / this.fps && this.current) pick = null;
    }
    if (pick && pick !== this.current) this.show(pick);
    if (this.ended && this.queue.length === 1 && this.current === this.queue[0]) {
      this.finish();
    }
  }

  show(pick) {
    if (this.decoding) return;
    this.current = pick;
    this.decoding = true;
    Promise.all(pick.blobs.map((b) => createImageBitmap(b)))
      .then((bitmaps) => this.onFrame(pick.header, bitmaps))
      .catch((err) => console.warn(err))
      .finally(() => {
        this.decoding = false;
      });
  }

  finish() {
    if (this._finished) return;
    this._finished = true;
    this.onEnd();
  }
}

function concat(a, b) {
  if (!a.length) return b;
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}
