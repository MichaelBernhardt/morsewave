import type { ReactNode } from "react";
import { Info } from "lucide-react";

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
}
export function Toggle({ checked, onChange, label, description, disabled = false }: ToggleProps) {
  return (
    <label className={`toggle-row ${disabled ? "is-disabled" : ""}`}>
      <span className="toggle-copy">
        <span>{label}</span>
        {description && <small>{description}</small>}
      </span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        disabled={disabled}
      />
      <span className="switch" aria-hidden="true"><span /></span>
    </label>
  );
}

interface RangeControlProps {
  label: string;
  value: number;
  minimum: number;
  maximum: number;
  step?: number;
  unit?: string;
  hint?: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}

export function RangeControl({
  label,
  value,
  minimum,
  maximum,
  step = 1,
  unit = "",
  hint,
  disabled = false,
  onChange,
}: RangeControlProps) {
  const percentage = (value - minimum) / (maximum - minimum) * 100;
  return (
    <div className={`range-control ${disabled ? "is-disabled" : ""}`}>
      <div className="range-heading">
        <label>
          {label}
          {hint && <span className="hint-icon" title={hint}><Info size={12} /></span>}
        </label>
        <output>{Math.round(value)}<span>{unit}</span></output>
      </div>
      <input
        type="range"
        min={minimum}
        max={maximum}
        step={step}
        value={value}
        disabled={disabled}
        style={{ "--range-progress": `${percentage}%` } as React.CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

interface PanelProps {
  eyebrow?: string;
  title: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Panel({ eyebrow, title, action, children, className = "" }: PanelProps) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-heading">
        <div>
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h2>{title}</h2>
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}
