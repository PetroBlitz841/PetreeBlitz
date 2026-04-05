import { Box, Typography, Alert } from "@mui/material";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { SpeciesBreakdownEntry } from "../../types";

interface SpeciesDistributionProps {
  species: SpeciesBreakdownEntry[];
}

export default function SpeciesDistribution({
  species,
}: SpeciesDistributionProps) {
  if (species.length === 0) {
    return (
      <Box>
        <Typography variant="h6" fontWeight={700} gutterBottom>
          Species Distribution
        </Typography>
        <Alert severity="info">
          No species data yet. Species appear as samples are identified and
          confirmed.
        </Alert>
      </Box>
    );
  }

  const data = species.map((s) => ({
    ...s,
    displayName: s.name.replace(/_/g, " "),
  }));

  return (
    <Box>
      <Typography variant="h6" fontWeight={700} gutterBottom>
        Species Distribution
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={2}>
        Top species by number of confirmed samples
      </Typography>
      <ResponsiveContainer
        width="100%"
        height={Math.max(250, data.length * 32)}
      >
        <BarChart data={data} layout="vertical" margin={{ left: 160 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="displayName"
            tick={{ fontSize: 11, fontStyle: "italic" }}
            width={150}
          />
          <Tooltip
            formatter={(value) => [`${value} samples`, "Samples"]}
          />
          <Bar dataKey="samples" fill="#4caf50" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
}
