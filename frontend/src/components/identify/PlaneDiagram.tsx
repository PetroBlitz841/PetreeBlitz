import { Plane } from "../../types";

interface PlaneDiagramProps {
  plane: Plane;
  size?: number;
}

export default function PlaneDiagram({ plane, size = 80 }: PlaneDiagramProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label={`${plane} cut diagram`}
    >
      {plane === "traverse" && <TransverseDiagram />}
      {plane === "radialLongitudinal" && <RadialLongitudinalDiagram />}
      {plane === "tangentialLongitudinal" && <TangentialLongitudinalDiagram />}
    </svg>
  );
}

/** Transverse: end-grain view — concentric rings with a highlighted cross-section face */
function TransverseDiagram() {
  return (
    <>
      {/* Log end face */}
      <ellipse
        cx="40"
        cy="40"
        rx="32"
        ry="32"
        fill="#c8a97e"
        stroke="#7a5230"
        strokeWidth="1.5"
      />
      {/* Growth rings */}
      {[26, 20, 14, 8].map((r) => (
        <ellipse
          key={r}
          cx="40"
          cy="40"
          rx={r}
          ry={r}
          fill="none"
          stroke="#7a5230"
          strokeWidth="0.8"
          opacity="0.6"
        />
      ))}
      {/* Rays */}
      {[0, 45, 90, 135].map((angle) => {
        const rad = (angle * Math.PI) / 180;
        return (
          <line
            key={angle}
            x1={40 + Math.cos(rad) * 4}
            y1={40 + Math.sin(rad) * 4}
            x2={40 + Math.cos(rad) * 30}
            y2={40 + Math.sin(rad) * 30}
            stroke="#7a5230"
            strokeWidth="0.6"
            opacity="0.4"
          />
        );
      })}
      {/* Highlight cut plane as a horizontal slice line */}
      <line
        x1="8"
        y1="40"
        x2="72"
        y2="40"
        stroke="#e05c2a"
        strokeWidth="2"
        strokeDasharray="4 2"
      />
    </>
  );
}

/** Radial Longitudinal: side view, cut along the radius (through the centre) */
function RadialLongitudinalDiagram() {
  return (
    <>
      {/* Log body */}
      <rect
        x="14"
        y="12"
        width="52"
        height="52"
        rx="3"
        fill="#c8a97e"
        stroke="#7a5230"
        strokeWidth="1.5"
      />
      {/* Horizontal grain lines */}
      {[20, 27, 34, 41, 48, 55].map((y) => (
        <line
          key={y}
          x1="14"
          y1={y}
          x2="66"
          y2={y}
          stroke="#7a5230"
          strokeWidth="0.7"
          opacity="0.5"
        />
      ))}
      {/* Vertical rays */}
      {[26, 33, 40, 47, 54].map((x) => (
        <line
          key={x}
          x1={x}
          y1="12"
          x2={x}
          y2="64"
          stroke="#7a5230"
          strokeWidth="0.5"
          opacity="0.3"
        />
      ))}
      {/* Highlight: vertical cut plane through centre */}
      <line
        x1="40"
        y1="8"
        x2="40"
        y2="72"
        stroke="#e05c2a"
        strokeWidth="2"
        strokeDasharray="4 2"
      />
    </>
  );
}

/** Tangential Longitudinal: side view, cut tangent to the growth rings */
function TangentialLongitudinalDiagram() {
  return (
    <>
      {/* Log body */}
      <rect
        x="14"
        y="12"
        width="52"
        height="52"
        rx="3"
        fill="#c8a97e"
        stroke="#7a5230"
        strokeWidth="1.5"
      />
      {/* Wavy grain lines (tangential gives the characteristic cathedral grain) */}
      {[
        "M14,22 Q27,18 40,22 Q53,26 66,22",
        "M14,31 Q27,27 40,31 Q53,35 66,31",
        "M14,40 Q27,36 40,40 Q53,44 66,40",
        "M14,49 Q27,45 40,49 Q53,53 66,49",
        "M14,58 Q27,54 40,58 Q53,62 66,58",
      ].map((d, i) => (
        <path
          key={i}
          d={d}
          stroke="#7a5230"
          strokeWidth="0.8"
          fill="none"
          opacity="0.55"
        />
      ))}
      {/* Highlight: vertical cut offset from centre (tangential) */}
      <line
        x1="52"
        y1="8"
        x2="52"
        y2="72"
        stroke="#e05c2a"
        strokeWidth="2"
        strokeDasharray="4 2"
      />
    </>
  );
}
