import { useEffect, useRef } from "react";
import type { AnalysisFrame } from "../audio/AudioEngine";

interface SignalScopeProps {
  frame: AnalysisFrame | null;
  active: boolean;
  minimumFrequency?: number;
  maximumFrequency?: number;
}

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(maximum, Math.max(minimum, value));

export function SignalScope({
  frame,
  active,
  minimumFrequency = 200,
  maximumFrequency = 4000,
}: SignalScopeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<Float32Array[]>([]);
  const frameRef = useRef<AnalysisFrame | null>(frame);
  const animationRef = useRef(0);

  useEffect(() => {
    frameRef.current = frame;
    if (frame?.spectrum) {
      historyRef.current.push(new Float32Array(frame.spectrum));
      if (historyRef.current.length > 54) historyRef.current.shift();
    }
  }, [frame]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();

    const draw = (now: number) => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const current = frameRef.current;
      context.clearRect(0, 0, width, height);

      const background = context.createLinearGradient(0, 0, 0, height);
      background.addColorStop(0, "rgba(8, 14, 18, 0.18)");
      background.addColorStop(1, "rgba(4, 8, 11, 0.7)");
      context.fillStyle = background;
      context.fillRect(0, 0, width, height);

      context.lineWidth = 1;
      context.strokeStyle = "rgba(157, 177, 184, 0.09)";
      for (let column = 0; column <= 6; column += 1) {
        const x = Math.round(width * column / 6) + 0.5;
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }
      for (let row = 1; row < 4; row += 1) {
        const y = Math.round(height * row / 4) + 0.5;
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }

      const history = historyRef.current;
      history.forEach((spectrum, historyIndex) => {
        const age = (historyIndex + 1) / Math.max(1, history.length);
        const baseline = height - 16 - (1 - age) * height * 0.52;
        context.beginPath();
        spectrum.forEach((db, index) => {
          const x = index / Math.max(1, spectrum.length - 1) * width;
          const normalized = clamp((db + 100) / 72, 0, 1);
          const y = baseline - normalized * (22 + age * 42);
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.strokeStyle = `rgba(73, 218, 178, ${0.025 + age * 0.12})`;
        context.lineWidth = age > 0.94 ? 1.6 : 1;
        context.stroke();
      });

      const spectrum = current?.spectrum ?? history.at(-1);
      if (spectrum) {
        const line = new Path2D();
        const fill = new Path2D();
        spectrum.forEach((db, index) => {
          const x = index / Math.max(1, spectrum.length - 1) * width;
          const normalized = clamp((db + 100) / 78, 0, 1);
          const y = height - 17 - normalized * (height * 0.63);
          if (index === 0) {
            line.moveTo(x, y);
            fill.moveTo(x, height);
            fill.lineTo(x, y);
          } else {
            line.lineTo(x, y);
            fill.lineTo(x, y);
          }
          if (index === spectrum.length - 1) {
            fill.lineTo(x, height);
            fill.closePath();
          }
        });
        const area = context.createLinearGradient(0, height * 0.3, 0, height);
        area.addColorStop(0, "rgba(68, 226, 180, 0.18)");
        area.addColorStop(1, "rgba(68, 226, 180, 0.015)");
        context.fillStyle = area;
        context.fill(fill);
        context.strokeStyle = "rgba(91, 235, 194, 0.9)";
        context.shadowColor = "rgba(59, 224, 177, 0.42)";
        context.shadowBlur = 8;
        context.lineWidth = 1.5;
        context.stroke(line);
        context.shadowBlur = 0;
      }

      if (current) {
        const tuneX = clamp(
          (current.frequency - minimumFrequency) / (maximumFrequency - minimumFrequency),
          0,
          1,
        ) * width;
        const pulse = reducedMotion ? 0.7 : 0.58 + Math.sin(now / 280) * 0.12;
        context.strokeStyle = current.metrics.gateOpen
          ? `rgba(255, 188, 91, ${pulse})`
          : "rgba(255, 188, 91, 0.34)";
        context.setLineDash([3, 5]);
        context.beginPath();
        context.moveTo(tuneX + 0.5, 0);
        context.lineTo(tuneX + 0.5, height);
        context.stroke();
        context.setLineDash([]);
        if (current.metrics.gateOpen) {
          const glow = context.createRadialGradient(tuneX, height * 0.36, 0, tuneX, height * 0.36, 70);
          glow.addColorStop(0, "rgba(255, 177, 64, 0.2)");
          glow.addColorStop(1, "rgba(255, 177, 64, 0)");
          context.fillStyle = glow;
          context.fillRect(tuneX - 70, 0, 140, height);
        }
      }

      if (!active && history.length === 0) {
        const ghostY = height * 0.64;
        context.beginPath();
        for (let x = 0; x <= width; x += 4) {
          const y = ghostY + Math.sin(x * 0.035) * 2 + Math.sin(x * 0.11) * 1.2;
          if (x === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
        context.strokeStyle = "rgba(135, 157, 165, 0.16)";
        context.stroke();
      }

      animationRef.current = requestAnimationFrame(draw);
    };
    animationRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animationRef.current);
      resizeObserver.disconnect();
    };
  }, [active, maximumFrequency, minimumFrequency]);

  return (
    <div className="signal-scope" role="img" aria-label="Live CW frequency spectrum from 200 to 4000 hertz">
      <canvas ref={canvasRef} />
      <div className="scope-frequency-labels" aria-hidden="true">
        <span>200</span><span>800</span><span>1400</span><span>2000</span><span>2600</span><span>3200</span><span>4000 Hz</span>
      </div>
      <div
        className={`scope-reticle ${frame?.metrics.gateOpen ? "is-keyed" : ""}`}
        style={{ left: `${clamp(((frame?.frequency ?? 700) - minimumFrequency) / (maximumFrequency - minimumFrequency), 0, 1) * 100}%` }}
        aria-hidden="true"
      >
        <span>{Math.round(frame?.frequency ?? 700)} Hz</span>
      </div>
    </div>
  );
}
