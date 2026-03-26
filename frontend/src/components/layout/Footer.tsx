import { Box, Typography } from "@mui/material";

export default function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        backgroundColor: "white",
        borderTop: 1,
        borderColor: "grey.200",
        py: 2,
        mt: "auto",
      }}
    >
      <Box sx={{ maxWidth: 1200, mx: "auto", px: 2, textAlign: "center" }}>
        <Typography variant="body2" color="text.secondary">
          PetreeBlitz - Powered by AI for Archaeobotany Research
        </Typography>
      </Box>
    </Box>
  );
}
