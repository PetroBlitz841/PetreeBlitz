import { Box, Typography, Alert } from "@mui/material";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { TimelineEntry } from "../../types";

interface GrowthTimelineProps {
  timeline: TimelineEntry[];
}

export default function GrowthTimeline({ timeline }: GrowthTimelineProps) {
  if (timeline.length === 0) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={700} gutterBottom>
          Growth Timeline
        </Typography>
        <Alert severity="info">
          No activity yet. Identifications and feedback will appear here over
          time.
        </Alert>
      </Box>
    );
  }

  // Build cumulative series
  const data = timeline.reduce<
    { date: string; Identifications: number; Feedback: number }[]
  >((acc, t) => {
    const prev =
      acc.length > 0
        ? acc[acc.length - 1]
        : { Identifications: 0, Feedback: 0 };
    acc.push({
      date: t.date,
      Identifications: prev.Identifications + t.identifications,
      Feedback: prev.Feedback + t.feedback,
    });
    return acc;
  }, []);

  return (
    <Box>
      <Typography variant="h6" fontWeight={700} gutterBottom>
        Growth Timeline
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={2}>
        Cumulative identifications and feedback submissions over time
      </Typography>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorId" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#4caf50" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#4caf50" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorFb" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#1565c0" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#1565c0" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          <Area
            type="monotone"
            dataKey="Identifications"
            stroke="#4caf50"
            fill="url(#colorId)"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="Feedback"
            stroke="#1565c0"
            fill="url(#colorFb)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
}
