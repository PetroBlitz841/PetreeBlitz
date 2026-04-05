import { Card, CardContent, Stack, Typography, Box } from "@mui/material";
import { ReactNode } from "react";

interface StatCardProps {
  icon: ReactNode;
  label: string;
  value: string | number;
  subtitle?: string;
  gradient?: [string, string];
}

export default function StatCard({
  icon,
  label,
  value,
  subtitle,
  gradient = ["#4caf50", "#2e7d32"],
}: StatCardProps) {
  return (
    <Card
      sx={{
        height: "100%",
        borderRadius: 2,
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* Accent gradient top strip */}
      <Box
        sx={{
          height: 4,
          background: `linear-gradient(90deg, ${gradient[0]}, ${gradient[1]})`,
        }}
      />
      <CardContent sx={{ py: 2 }}>
        <Stack direction="row" alignItems="center" spacing={1.5}>
          <Box
            sx={{
              width: 44,
              height: 44,
              borderRadius: 1.5,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: `linear-gradient(135deg, ${gradient[0]}22, ${gradient[1]}22)`,
              color: gradient[0],
              flexShrink: 0,
            }}
          >
            {icon}
          </Box>
          <Stack spacing={0}>
            <Typography
              variant="caption"
              color="text.secondary"
              fontWeight={500}
              lineHeight={1.2}
            >
              {label}
            </Typography>
            <Typography variant="h5" fontWeight={700} lineHeight={1.2}>
              {value}
            </Typography>
            {subtitle && (
              <Typography
                variant="caption"
                color="text.secondary"
                lineHeight={1.2}
              >
                {subtitle}
              </Typography>
            )}
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
