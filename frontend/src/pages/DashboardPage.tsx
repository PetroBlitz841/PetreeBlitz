import { useEffect, useState, useCallback } from "react";
import {
  Box,
  Typography,
  Stack,
  Alert,
  CircularProgress,
  Divider,
  Paper,
} from "@mui/material";
import api from "../services/api";
import { DashboardStats } from "../types";
import OverviewSection from "../components/dashboard/OverviewSection";
import GrowthTimeline from "../components/dashboard/GrowthTimeline";
import AccuracySection from "../components/dashboard/AccuracySection";
import FeatureAnalyticsSection from "../components/dashboard/FeatureAnalyticsSection";
import SpeciesDistribution from "../components/dashboard/SpeciesDistribution";
import FederatedSection from "../components/dashboard/FederatedSection";

let statsCache: DashboardStats | null = null;

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(statsCache);
  const [loading, setLoading] = useState(statsCache === null);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    if (statsCache === null) setLoading(true);
    setError(null);
    try {
      const resp = await api.get<DashboardStats>("/stats");
      statsCache = resp.data;
      setStats(resp.data);
    } catch (err) {
      console.error(err);
      if (statsCache === null)
        setError("Failed to load dashboard stats. Please try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return (
    <Box
      sx={{
        flex: 1,
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        width: "100%",
        px: 2,
        py: 4,
      }}
    >
      <Box sx={{ width: "100%", maxWidth: 1200 }}>
        {/* Header */}
        <Stack direction="column" spacing={0.5} mb={4}>
          <Typography variant="h4" color="primary" fontWeight="bold">
            Model Learning Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Track how the identification model grows through expert feedback and
            collaborative learning
          </Typography>
        </Stack>

        {loading && (
          <Box sx={{ display: "flex", justifyContent: "center", py: 8 }}>
            <CircularProgress />
          </Box>
        )}

        {error && !loading && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {stats && !loading && (
          <Stack spacing={4}>
            {/* 1. Overview KPIs */}
            <OverviewSection
              stats={stats.overview}
              accuracy={stats.accuracy.rate}
            />

            <Divider />

            {/* 2. Growth timeline + Accuracy side by side on larger screens */}
            <Stack direction={{ xs: "column", md: "row" }} spacing={3}>
              <Paper sx={{ flex: 2, p: 3, borderRadius: 2 }} variant="outlined">
                <GrowthTimeline timeline={stats.timeline} />
              </Paper>
              <Paper sx={{ flex: 1, p: 3, borderRadius: 2 }} variant="outlined">
                <AccuracySection accuracy={stats.accuracy} />
              </Paper>
            </Stack>

            <Divider />

            {/* 3. Species distribution + Feature analytics */}
            <Stack direction={{ xs: "column", md: "row" }} spacing={3}>
              <Paper sx={{ flex: 1, p: 3, borderRadius: 2 }} variant="outlined">
                <SpeciesDistribution species={stats.species_breakdown} />
              </Paper>
              <Paper sx={{ flex: 1, p: 3, borderRadius: 2 }} variant="outlined">
                <FeatureAnalyticsSection features={stats.feature_analytics} />
              </Paper>
            </Stack>

            <Divider />

            {/* 4. Federated learning */}
            <Paper sx={{ p: 3, borderRadius: 2 }} variant="outlined">
              <FederatedSection federated={stats.federated} />
            </Paper>
          </Stack>
        )}
      </Box>
    </Box>
  );
}
