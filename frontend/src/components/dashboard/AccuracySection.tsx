import { Box, Typography, Stack, Alert } from "@mui/material";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from "recharts";
import { AccuracyStats } from "../../types";

const COLORS = ["#4caf50", "#f44336"];

interface AccuracySectionProps {
  accuracy: AccuracyStats;
}

export default function AccuracySection({ accuracy }: AccuracySectionProps) {
  const total = accuracy.correct + accuracy.incorrect;

  if (total === 0) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={700} gutterBottom>
          Prediction Accuracy
        </Typography>
        <Alert severity="info">
          No feedback submitted yet. Accuracy data will appear once experts
          start reviewing predictions.
        </Alert>
      </Box>
    );
  }

  const data = [
    { name: "Correct", value: accuracy.correct },
    { name: "Incorrect", value: accuracy.incorrect },
  ];

  return (
    <Box>
      <Typography variant="h6" fontWeight={700} gutterBottom>
        Prediction Accuracy
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={2}>
        How often expert feedback confirms the model's top prediction
      </Typography>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={3}
        alignItems="center"
      >
        <ResponsiveContainer width={220} height={220}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={55}
              outerRadius={90}
              paddingAngle={3}
              dataKey="value"
              label={({ name, percent }: { name?: string; percent?: number }) =>
                `${name ?? ""} ${((percent ?? 0) * 100).toFixed(0)}%`
              }
              labelLine={false}
            >
              {data.map((_entry, i) => (
                <Cell key={i} fill={COLORS[i]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>

        <Stack spacing={1}>
          <Typography variant="h3" fontWeight={700} color="primary.main">
            {accuracy.rate != null
              ? `${(accuracy.rate * 100).toFixed(1)}%`
              : "—"}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            overall accuracy rate
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {accuracy.correct} correct · {accuracy.incorrect} corrected ·{" "}
            {total} total reviews
          </Typography>
        </Stack>
      </Stack>
    </Box>
  );
}
