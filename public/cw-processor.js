/* Morsewave narrow-band analysis processor.
 * Runs off the UI thread and emits a timing sample every 256 audio samples.
 */
class CwProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.hopSize = 256;
    this.hopIndex = 0;
    this.analysisSize = 1024;
    this.analysisRing = new Float32Array(this.analysisSize);
    this.analysisBuffer = new Float32Array(this.analysisSize);
    this.analysisWindow = this.makeWindow(this.analysisSize);
    this.analysisWindowSum = this.analysisWindow.reduce((sum, value) => sum + value, 0);
    this.analysisIndex = 0;
    this.analysisCount = 0;
    this.tuneSize = 4096;
    this.tuneRing = new Float32Array(this.tuneSize);
    this.tuneBuffer = new Float32Array(this.tuneSize);
    this.tuneWindow = this.makeWindow(this.tuneSize);
    this.tuneWindowSum = this.tuneWindow.reduce((sum, value) => sum + value, 0);
    this.tuneIndex = 0;
    this.tuneCount = 0;
    this.frameCount = 0;
    this.targetFrequency = 700;
    this.autoTune = true;
    this.minimumFrequency = 200;
    this.maximumFrequency = 4000;
    this.peakProminenceDb = 0;
    this.carrierLocked = false;
    this.candidateFrequency = this.targetFrequency;
    this.candidateHits = 0;
    this.carrierQuietScans = 0;
    this.tuneLowEnvelope = null;
    this.tuneHighEnvelope = null;
    this.tunePreviousMagnitude = null;
    this.tuneKeyingActivity = null;
    this.tuneHistoryFrames = 0;
    this.lastSpectrum = null;
    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type !== "config") return;
      if (Number.isFinite(data.targetFrequency)) this.targetFrequency = data.targetFrequency;
      if (typeof data.autoTune === "boolean") this.autoTune = data.autoTune;
      if (Number.isFinite(data.minimumFrequency)) this.minimumFrequency = data.minimumFrequency;
      if (Number.isFinite(data.maximumFrequency)) this.maximumFrequency = data.maximumFrequency;
    };
  }

  makeWindow(size) {
    return Float32Array.from({ length: size }, (_, index) =>
      0.5 - 0.5 * Math.cos(2 * Math.PI * index / (size - 1))
    );
  }

  magnitudeAt(buffer, frequency) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const coefficient = 2 * Math.cos(omega);
    let previous = 0;
    let previousTwo = 0;
    const windowValues = buffer.length === this.tuneSize ? this.tuneWindow : this.analysisWindow;
    const windowSum = buffer.length === this.tuneSize ? this.tuneWindowSum : this.analysisWindowSum;
    for (let index = 0; index < buffer.length; index += 1) {
      const value = buffer[index] * windowValues[index] + coefficient * previous - previousTwo;
      previousTwo = previous;
      previous = value;
    }
    const real = previous - previousTwo * Math.cos(omega);
    const imaginary = previousTwo * Math.sin(omega);
    return 2 * Math.hypot(real, imaginary) / Math.max(1, windowSum);
  }

  orderedTuneBuffer() {
    for (let index = 0; index < this.tuneSize; index += 1) {
      this.tuneBuffer[index] = this.tuneRing[(this.tuneIndex + index) % this.tuneSize];
    }
    return this.tuneBuffer;
  }

  orderedAnalysisBuffer() {
    for (let index = 0; index < this.analysisSize; index += 1) {
      this.analysisBuffer[index] = this.analysisRing[(this.analysisIndex + index) % this.analysisSize];
    }
    return this.analysisBuffer;
  }

  scanSpectrum() {
    if (this.tuneCount < this.tuneSize) return;
    const buffer = this.orderedTuneBuffer();
    // A coarse full-IF sweep keeps processor cost bounded even across a wide
    // receiver passband. The winning bin is then refined to 5 Hz resolution.
    const step = 30;
    const magnitudes = [];
    let bestMagnitude = 0;
    let bestFrequency = this.targetFrequency;
    for (let frequency = this.minimumFrequency; frequency <= this.maximumFrequency; frequency += step) {
      const magnitude = this.magnitudeAt(buffer, frequency);
      magnitudes.push(magnitude);
      if (magnitude > bestMagnitude) {
        bestMagnitude = magnitude;
        bestFrequency = frequency;
      }
    }

    for (let frequency = Math.max(this.minimumFrequency, bestFrequency - step);
      frequency <= Math.min(this.maximumFrequency, bestFrequency + step);
      frequency += 5) {
      const magnitude = this.magnitudeAt(buffer, frequency);
      if (magnitude > bestMagnitude) {
        bestMagnitude = magnitude;
        bestFrequency = frequency;
      }
    }

    if (!this.tuneLowEnvelope || this.tuneLowEnvelope.length !== magnitudes.length) {
      this.tuneLowEnvelope = Float32Array.from(magnitudes);
      this.tuneHighEnvelope = Float32Array.from(magnitudes);
      this.tunePreviousMagnitude = Float32Array.from(magnitudes);
      this.tuneKeyingActivity = new Float32Array(magnitudes.length);
      this.tuneHistoryFrames = 1;
    } else {
      for (let index = 0; index < magnitudes.length; index += 1) {
        const magnitude = magnitudes[index];
        const low = this.tuneLowEnvelope[index];
        const high = this.tuneHighEnvelope[index];
        const previous = this.tunePreviousMagnitude[index] || 1e-8;
        const transitionDb = Math.min(32, Math.abs(20 * Math.log10((magnitude + 1e-8) / (previous + 1e-8))));
        this.tuneLowEnvelope[index] += (magnitude - low) * (magnitude < low ? 0.34 : 0.012);
        this.tuneHighEnvelope[index] += (magnitude - high) * (magnitude > high ? 0.34 : 0.022);
        this.tuneKeyingActivity[index] = this.tuneKeyingActivity[index] * 0.78 + transitionDb * 0.22;
        this.tunePreviousMagnitude[index] = magnitude;
      }
      this.tuneHistoryFrames += 1;
    }

    // Strong receiver birdies and tune-up carriers can dwarf the wanted CW in
    // raw energy. Rank coarse bins by both peak energy and on/off modulation:
    // a keyed carrier develops separated high/low envelopes, while a steady
    // heterodyne gets no modulation bonus. Persistence below still rejects
    // one-off static crashes.
    let bestIndex = Math.round((bestFrequency - this.minimumFrequency) / step);
    let bestSelectionScoreDb = -Infinity;
    let bestModulationDb = 0;
    if (this.tuneHistoryFrames >= 4) {
      for (let index = 0; index < magnitudes.length; index += 1) {
        const peak = this.tuneHighEnvelope[index] || 1e-8;
        const peakDb = 20 * Math.log10(peak + 1e-8);
        const modulationDb = this.tuneKeyingActivity[index] || 0;
        const modulationBonus = Math.min(26, Math.max(0, modulationDb - 1.5) * 1.35);
        // While seeking, treat the configured center as a soft preference, not
        // a narrow filter. This resolves two equally valid keyed stations in a
        // passband without preventing acquisition elsewhere when only one is
        // present. Once locked, continuity logic below owns the decision.
        const binFrequency = this.minimumFrequency + index * step;
        const acquisitionPenaltyDb = this.carrierLocked
          ? 0
          : Math.min(16, Math.abs(binFrequency - this.targetFrequency) / 80);
        const scoreDb = peakDb + modulationBonus - acquisitionPenaltyDb;
        if (scoreDb > bestSelectionScoreDb) {
          bestSelectionScoreDb = scoreDb;
          bestIndex = index;
          bestModulationDb = modulationDb;
        }
      }
      bestFrequency = this.minimumFrequency + bestIndex * step;
      bestMagnitude = magnitudes[bestIndex] || 0;
      const selectedFloor = this.tuneLowEnvelope[bestIndex] || 1e-8;
      // Refine only while the selected keyed carrier is actually up. During a
      // key-up interval the coarse center is safer than a random local noise bin.
      if (bestMagnitude > selectedFloor * 1.45 || bestModulationDb < 3) {
        for (let frequency = Math.max(this.minimumFrequency, bestFrequency - step);
          frequency <= Math.min(this.maximumFrequency, bestFrequency + step);
          frequency += 5) {
          const magnitude = this.magnitudeAt(buffer, frequency);
          if (magnitude > bestMagnitude) {
            bestMagnitude = magnitude;
            bestFrequency = frequency;
          }
        }
      }
    }

    const ordered = [...magnitudes].sort((a, b) => a - b);
    const baseline = ordered[Math.floor(ordered.length * 0.58)] || 1e-8;
    const peakEnvelope = this.tuneHighEnvelope?.[bestIndex] || bestMagnitude;
    const orderedPeaks = this.tuneHighEnvelope ? [...this.tuneHighEnvelope].sort((a, b) => a - b) : ordered;
    const peakBaseline = orderedPeaks[Math.floor(orderedPeaks.length * 0.58)] || baseline;
    const globalProminenceDb = 20 * Math.log10((peakEnvelope + 1e-8) / (peakBaseline + 1e-8));
    const bestDb = 20 * Math.log10(peakEnvelope + 1e-8);

    // Once acquired, prefer a carrier near the current IF even when a stronger
    // station appears elsewhere. During key-up intervals the tuned frequency is
    // held; broadband noise peaks must persist at one frequency before they can
    // pull the receiver away from a valid CW signal.
    let localMagnitude = 0;
    let localFrequency = this.targetFrequency;
    for (let frequency = Math.max(this.minimumFrequency, this.targetFrequency - 70);
      frequency <= Math.min(this.maximumFrequency, this.targetFrequency + 70);
      frequency += 10) {
      const magnitude = this.magnitudeAt(buffer, frequency);
      if (magnitude > localMagnitude) {
        localMagnitude = magnitude;
        localFrequency = frequency;
      }
    }
    const localProminenceDb = 20 * Math.log10((localMagnitude + 1e-8) / (baseline + 1e-8));
    const localDb = 20 * Math.log10(localMagnitude + 1e-8);
    const localCarrier = this.carrierLocked && localProminenceDb > 8 && localDb > -84;
    const localIndex = Math.max(0, Math.min(magnitudes.length - 1,
      Math.round((this.targetFrequency - this.minimumFrequency) / step)));
    const localPeak = this.tuneHighEnvelope?.[localIndex] || localMagnitude;
    const localModulationDb = this.tuneKeyingActivity?.[localIndex] || 0;
    const localSelectionScoreDb = 20 * Math.log10(localPeak + 1e-8) +
      Math.min(26, Math.max(0, localModulationDb - 1.5) * 1.35);
    // A receiver must not splice two simultaneous CW conversations merely
    // because the other station becomes stronger during a word gap. Keying
    // transitions refresh the tenure of the current carrier. A distant
    // replacement is allowed only after the track has shown no keying for a
    // sustained interval; nearby candidates remain eligible so ordinary
    // oscillator drift can still be followed continuously.
    if (!this.carrierLocked || localModulationDb > 2.4) {
      this.carrierQuietScans = 0;
    } else {
      this.carrierQuietScans += 1;
    }
    const releaseScans = Math.max(6, Math.ceil(3.5 * sampleRate / (this.hopSize * 9)));
    const keyedAlternative = this.tuneHistoryFrames >= 4 && bestModulationDb > 5.5 &&
      localModulationDb < 3.2 &&
      Math.abs(bestFrequency - this.targetFrequency) > 70 &&
      bestSelectionScoreDb > localSelectionScoreDb + 2.5;
    const strongCarrier = globalProminenceDb > 13 && bestDb > -84;
    const moderateCarrier = globalProminenceDb > 8 && bestDb > -54;

    if (this.autoTune && localCarrier && !keyedAlternative) {
      this.targetFrequency += (localFrequency - this.targetFrequency) * 0.28;
      this.candidateHits = 0;
      this.peakProminenceDb = localProminenceDb;
    } else if (this.autoTune && (strongCarrier || moderateCarrier)) {
      if (Math.abs(bestFrequency - this.candidateFrequency) <= 35) {
        this.candidateFrequency += (bestFrequency - this.candidateFrequency) * 0.45;
        this.candidateHits += strongCarrier ? 2 : 1;
      } else {
        this.candidateFrequency = bestFrequency;
        this.candidateHits = strongCarrier ? 2 : 1;
      }
      // Do not lock on the first loud FFT bin. At startup the modulation
      // envelopes have not separated yet, so an impulse or window sidelobe can
      // otherwise win permanently. Two stable, modulation-ranked scans are
      // enough to acquire without materially clipping the first character.
      const canAcquire = !this.carrierLocked && this.tuneHistoryFrames >= 4 &&
        this.candidateHits >= 4;
      const nearbyCandidate = Math.abs(this.candidateFrequency - this.targetFrequency) <= 90;
      const canReplace = this.carrierLocked && strongCarrier && this.candidateHits >= 6 &&
        (nearbyCandidate || this.carrierQuietScans >= releaseScans);
      if (canAcquire || canReplace) {
        this.targetFrequency = this.candidateFrequency;
        this.carrierLocked = true;
        this.candidateHits = 0;
        this.carrierQuietScans = 0;
      }
      this.peakProminenceDb = globalProminenceDb;
    } else {
      this.candidateHits = Math.max(0, this.candidateHits - 1);
      this.peakProminenceDb = this.carrierLocked ? localProminenceDb : globalProminenceDb;
    }
    this.targetFrequency = Math.max(this.minimumFrequency, Math.min(this.maximumFrequency, this.targetFrequency));
    this.lastSpectrum = Float32Array.from(magnitudes, (magnitude) =>
      Math.max(-110, Math.min(0, 20 * Math.log10(magnitude + 1e-8))),
    );
  }

  analyzeFrame(buffer) {
    this.frameCount += 1;
    const amplitude = this.magnitudeAt(buffer, this.targetFrequency);
    let squareSum = 0;
    for (let index = 0; index < buffer.length; index += 1) {
      squareSum += buffer[index] * buffer[index];
    }
    const rms = Math.sqrt(squareSum / buffer.length);
    const levelDb = Math.max(-120, 20 * Math.log10(amplitude + 1e-8));
    const broadbandDb = Math.max(-120, 20 * Math.log10(rms + 1e-8));
    const purity = Math.max(0, Math.min(1, amplitude * amplitude / Math.max(1e-10, 2 * rms * rms)));

    let spectrum;
    if (this.tuneCount >= this.tuneSize && (this.frameCount === 1 || this.frameCount % 9 === 0)) {
      this.scanSpectrum();
      spectrum = this.lastSpectrum;
      this.lastSpectrum = null;
    }

    this.port.postMessage({
      type: "analysis",
      timestamp: currentTime * 1000,
      levelDb,
      broadbandDb,
      purity,
      frequency: this.targetFrequency,
      prominenceDb: this.peakProminenceDb,
      spectrum,
    });
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    const blockLength = channels[0].length;
    for (let inputIndex = 0; inputIndex < blockLength; inputIndex += 1) {
      let sample = 0;
      for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
        sample += channels[channelIndex][inputIndex] || 0;
      }
      sample /= channels.length;
      this.analysisRing[this.analysisIndex] = sample;
      this.analysisIndex = (this.analysisIndex + 1) % this.analysisSize;
      this.analysisCount = Math.min(this.analysisSize, this.analysisCount + 1);
      this.hopIndex += 1;
      this.tuneRing[this.tuneIndex] = sample;
      this.tuneIndex = (this.tuneIndex + 1) % this.tuneSize;
      this.tuneCount = Math.min(this.tuneSize, this.tuneCount + 1);
      if (this.hopIndex === this.hopSize) {
        if (this.analysisCount >= this.analysisSize) this.analyzeFrame(this.orderedAnalysisBuffer());
        this.hopIndex = 0;
      }
    }
    return true;
  }
}

registerProcessor("cw-processor", CwProcessor);
