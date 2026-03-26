import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import * as d3 from "d3-geo";
import { RegionCode, L1_COLORS as L1_COLOR } from "../../utils/regions";

// ─── Constants ────────────────────────────────────────────────────────────────

const GEOJSON_URL =
  "https://raw.githubusercontent.com/tdwg/wgsrpd/master/geojson/level3.geojson";

const L1_LABELS: Record<string, string> = {
  "1": "Europe",
  "2": "Africa",
  "3": "Asia-Temperate",
  "4": "Asia-Tropical",
  "5": "Australasia",
  "6": "Pacific",
  "7": "N. America",
  "8": "S. America",
  "9": "Antarctic",
};

const SELECTED_COLOR = "#ff8800";
const HOVER_COLOR = "#ffcc33";
const SIBLING_COLOR = "#fdc314";

const DOT_STEP = 7; // spacing between dot centres (px)
const DOT_R = 2.5; // base dot radius (px)

// ─── GeoJSON cache (module-level) ─────────────────────────────────────────────

let geoCache: GeoJSON.FeatureCollection | null = null;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function codeProp(code: string): "LEVEL1_COD" | "LEVEL2_COD" | "LEVEL3_COD" {
  if (code.length === 1) return "LEVEL1_COD";
  if (code.length === 2) return "LEVEL2_COD";
  return "LEVEL3_COD";
}

/** True if a feature's properties match a given code at any level. */
function matchesCode(
  l1: string,
  l2: string,
  l3: string,
  code: string,
): boolean {
  const prop = codeProp(code);
  if (prop === "LEVEL1_COD") return l1 === code;
  if (prop === "LEVEL2_COD") return l2 === code;
  return l3 === code;
}

function withAlpha(hex: string, alpha: number): string {
  const a = Math.round(alpha * 255)
    .toString(16)
    .padStart(2, "0");
  return `${hex}${a}`;
}

function patternId(color: string): string {
  return `dot-${color.replace(/[^a-zA-Z0-9]/g, "")}`;
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface FeatureMeta {
  code: string; // L3 code
  name: string; // L3 name
  l1: string;
  l2: string;
  path: string;
}

interface TooltipState {
  x: number;
  y: number;
  text: string;
}

export interface TDWGDotMapProps {
  selectedCode: RegionCode | "";
  highlightCodes?: RegionCode[];
  onRegionClick: (code: RegionCode) => void;
  height: number;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function TDWGDotMap({
  selectedCode,
  highlightCodes = [],
  onRegionClick,
  height,
}: TDWGDotMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [geo, setGeo] = useState<GeoJSON.FeatureCollection | null>(geoCache);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hovered, setHovered] = useState("");
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [width, setWidth] = useState(720);

  // ── Track container width ──────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) =>
      setWidth(Math.floor(entries[0].contentRect.width)),
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── Load GeoJSON once ─────────────────────────────────────────────────────
  useEffect(() => {
    if (geoCache) return;

    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);

    fetch(GEOJSON_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<GeoJSON.FeatureCollection>;
      })
      .then((d) => {
        if (cancelled) return;
        geoCache = d;
        setGeo(d);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // ── Projection ────────────────────────────────────────────────────────────
  const proj = useMemo(
    () =>
      d3
        .geoNaturalEarth1()
        .scale(width / 6.28)
        .translate([width / 2, height / 2]),
    [width, height],
  );

  const pathGen = useMemo(() => d3.geoPath(proj), [proj]);

  // ── Feature list ──────────────────────────────────────────────────────────
  const features = useMemo((): FeatureMeta[] => {
    if (!geo) return [];
    return geo.features
      .map((f) => {
        const p = f.properties ?? {};
        return {
          code: String(p["LEVEL3_COD"] ?? ""),
          name: String(p["LEVEL3_NAM"] ?? ""),
          l1: String(p["LEVEL1_COD"] ?? ""),
          l2: String(p["LEVEL2_COD"] ?? ""),
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          path: pathGen(f as any) ?? "",
        };
      })
      .filter((f) => f.code && f.path);
  }, [geo, pathGen]);

  // ── Static paths ──────────────────────────────────────────────────────────
  const graticulePath = useMemo(
    () => pathGen(d3.geoGraticule()()) ?? undefined,
    [pathGen],
  );
  const spherePath = useMemo(
    () => pathGen({ type: "Sphere" } as never) ?? undefined,
    [pathGen],
  );

  // ── Colour map ────────────────────────────────────────────────────────────
  const colorMap = useMemo(() => {
    const dimmed = !!selectedCode;
    const highlightSet = new Set<string>(highlightCodes);
    const map: Record<string, string> = {};

    for (const f of features) {
      const isSelected =
        selectedCode !== "" && matchesCode(f.l1, f.l2, f.code, selectedCode);

      const isHighlighted = [...highlightSet].some((hc) =>
        matchesCode(f.l1, f.l2, f.code, hc),
      );

      if (isSelected) {
        map[f.code] = SELECTED_COLOR;
      } else if (isHighlighted) {
        map[f.code] = SIBLING_COLOR;
      } else {
        const base = L1_COLOR[f.l1] ?? "#888888";
        map[f.code] = dimmed ? withAlpha(base, 0.27) : base;
      }
    }
    return map;
  }, [features, selectedCode, highlightCodes]);

  // Hover overrides colour without rebuilding the full map
  const colorFor = useCallback(
    (code: string): string => {
      if (code === hovered && colorMap[code] !== SELECTED_COLOR)
        return HOVER_COLOR;
      return colorMap[code] ?? "#888888";
    },
    [colorMap, hovered],
  );

  // ── SVG pattern defs (one per unique colour, normal + active size) ────────
  const patternColors = useMemo(() => {
    const seen = new Set<string>(Object.values(colorMap));
    seen.add(HOVER_COLOR);
    seen.add(SELECTED_COLOR);
    seen.add(SIBLING_COLOR);
    return Array.from(seen);
  }, [colorMap]);

  // ── Click handler ─────────────────────────────────────────────────────────
  // Emit the code at the same granularity as selectedCode so the caller
  // stays at the level they're already browsing. Defaults to L3 when nothing
  // is selected yet.
  const handleRegionClick = useCallback(
    (f: FeatureMeta) => {
      let emitted: string;
      if (selectedCode.length === 1) {
        emitted = f.l1;
      } else if (selectedCode.length === 2) {
        emitted = f.l2;
      } else {
        emitted = f.code;
      }
      onRegionClick(emitted as RegionCode);
    },
    [onRegionClick, selectedCode],
  );

  // ── Hover handlers ────────────────────────────────────────────────────────
  const clearHover = useCallback(() => {
    setHovered("");
    setTooltip(null);
  }, []);

  const handleRegionHover = useCallback(
    (f: FeatureMeta, e: React.MouseEvent<SVGPathElement>) => {
      const rect = (
        e.currentTarget.closest("svg") as SVGSVGElement
      ).getBoundingClientRect();
      setHovered(f.code);
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        text: `${f.name} (${f.code})`,
      });
    },
    [],
  );

  // ── Render ────────────────────────────────────────────────────────────────
  if (error) {
    return (
      <div style={{ color: "#c0392b", padding: 8, fontFamily: "monospace" }}>
        ⚠ Failed to load map: {error}
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{ width: "100%", userSelect: "none" }}>
      {loading && (
        <div
          style={{
            textAlign: "center",
            padding: 20,
            color: "#666",
            fontFamily: "monospace",
            fontSize: 13,
          }}
        >
          Loading…
        </div>
      )}

      {!loading && geo && (
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          style={{ display: "block", borderRadius: 8 }}
          onMouseLeave={clearHover}
        >
          <defs>
            {/* Normal-size dot patterns */}
            {patternColors.map((color) => (
              <pattern
                key={color}
                id={patternId(color)}
                x={0}
                y={0}
                width={DOT_STEP}
                height={DOT_STEP}
                patternUnits="userSpaceOnUse"
              >
                <circle
                  cx={DOT_STEP / 2}
                  cy={DOT_STEP / 2}
                  r={DOT_R}
                  fill={color}
                />
              </pattern>
            ))}

            {/* Enlarged-dot patterns for active (selected / hovered) features */}
            {patternColors.map((color) => (
              <pattern
                key={`${color}-active`}
                id={`${patternId(color)}-active`}
                x={0}
                y={0}
                width={DOT_STEP}
                height={DOT_STEP}
                patternUnits="userSpaceOnUse"
              >
                <circle
                  cx={DOT_STEP / 2}
                  cy={DOT_STEP / 2}
                  r={DOT_R * 1.25}
                  fill={color}
                />
              </pattern>
            ))}

            {/* One clipPath per feature */}
            {features.map((f) => (
              <clipPath key={`cp-${f.code}`} id={`cp-${f.code}`}>
                <path d={f.path} />
              </clipPath>
            ))}
          </defs>

          {/* Sphere outline */}
          <path d={spherePath} fill="none" stroke="#aacde0" strokeWidth={0.8} />

          {/* Graticule */}
          <path
            d={graticulePath}
            fill="none"
            stroke="#b8d8e8"
            strokeWidth={0.3}
          />

          {/* Features */}
          {features.map((f) => {
            const isActive =
              f.code === hovered || colorMap[f.code] === SELECTED_COLOR;
            const color = colorFor(f.code);
            const pid = isActive
              ? `${patternId(color)}-active`
              : patternId(color);

            return (
              <g key={f.code}>
                {/* Dot fill clipped to feature polygon */}
                <rect
                  x={0}
                  y={0}
                  width={width}
                  height={height}
                  fill={`url(#${pid})`}
                  clipPath={`url(#cp-${f.code})`}
                  style={{ pointerEvents: "none" }}
                />
                {/* Transparent hit area */}
                <path
                  d={f.path}
                  fill="transparent"
                  stroke="none"
                  style={{ cursor: "pointer" }}
                  onClick={() => handleRegionClick(f)}
                  onMouseMove={(e) => handleRegionHover(f, e)}
                  onMouseLeave={clearHover}
                />
              </g>
            );
          })}

          {/* Tooltip */}
          {tooltip &&
            (() => {
              const PAD = 18;
              const tw = tooltip.text.length * 6.5 + PAD;
              const tx = Math.min(tooltip.x + 12, width - tw - 4);
              const ty = Math.max(tooltip.y - 34, 4);
              return (
                <g
                  transform={`translate(${tx},${ty})`}
                  style={{ pointerEvents: "none" }}
                >
                  <rect
                    width={tw}
                    height={22}
                    rx={4}
                    fill="rgba(10,20,30,0.82)"
                  />
                  <text
                    x={9}
                    y={15}
                    fontSize={11}
                    fill="white"
                    fontFamily="monospace"
                  >
                    {tooltip.text}
                  </text>
                </g>
              );
            })()}
        </svg>
      )}

      {/* Legend */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "6px 12px",
          marginTop: 8,
        }}
      >
        {Object.entries(L1_COLOR).map(([code, color]) => (
          <LegendDot key={code} color={color} label={L1_LABELS[code]} />
        ))}
        <LegendDot color={SELECTED_COLOR} label="Selected" />
        {highlightCodes.length > 0 && (
          <LegendDot color={SIBLING_COLOR} label="Related" />
        )}
      </div>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: color,
          display: "inline-block",
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: 10, color: "#555", fontFamily: "sans-serif" }}>
        {label}
      </span>
    </span>
  );
}
