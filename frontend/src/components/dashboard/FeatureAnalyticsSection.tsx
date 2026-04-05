import { Box, Typography, Alert } from "@mui/material";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";
import { FeatureAnalyticsEntry } from "../../types";

const CATEGORY_COLOR: Record<string, string> = {
  "Growth Rings": "#795548",
  Porosity: "#1565C0",
  Vessels: "#2E7D32",
  "Vessel Pits": "#558B2F",
  Fibers: "#F57F17",
  Parenchyma: "#6A1B9A",
  Rays: "#00838F",
  Crystals: "#AD1457",
  Unknown: "#757575",
};

interface FeatureAnalyticsSectionProps {
  features: FeatureAnalyticsEntry[];
}

export default function FeatureAnalyticsSection({
  features,
}: FeatureAnalyticsSectionProps) {
  if (features.length === 0) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={700} gutterBottom>
          Feature Correction Analytics
        </Typography>
        <Alert severity="info">
          No feature corrections yet. When experts adjust IAWA feature
          identifications, the most-corrected features will appear here.
        </Alert>
      </Box>
    );
  }

  const top10 = features.slice(0, 10);

  return (
    <Box>
      <Typography variant="h6" fontWeight={700} gutterBottom>
        Feature Correction Analytics
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={2}>
        Most-corrected IAWA features by expert feedback — color indicates
        anatomical category
      </Typography>
      <ResponsiveContainer
        width="100%"
        height={Math.max(250, top10.length * 36)}
      >
        <BarChart data={top10} layout="vertical" margin={{ left: 140 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 11 }}
            width={130}
          />
          <Tooltip
            formatter={(
              value,
              _name,
              props,
            ) => {
              const entry = (props as { payload: FeatureAnalyticsEntry }).payload;
              return [
                `${value} corrections (avg importance ${entry.avg_importance}×)`,
                entry.category,
              ];
            }}
          />
          <Bar dataKey="corrections" radius={[0, 4, 4, 0]}>
            {top10.map((f, i) => (
              <Cell
                key={i}
                fill={CATEGORY_COLOR[f.category] ?? CATEGORY_COLOR.Unknown}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
}
