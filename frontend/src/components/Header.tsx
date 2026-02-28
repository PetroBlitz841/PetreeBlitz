import { AppBar, Toolbar, Typography, Box, Tabs, Tab } from "@mui/material";
import CollectionsIcon from "@mui/icons-material/Collections";
import { Link, useLocation } from "react-router-dom";
import { Biotech } from "@mui/icons-material";

const navItems = [
  { label: "Identify Trees", to: "/", icon: <Biotech fontSize="small" /> },
  { label: "Albums", to: "/albums", icon: <CollectionsIcon fontSize="small" /> },
];

export default function Header() {
  const location = useLocation();
  const current = navItems.find(item => item.to === location.pathname)?.to || "/";

  return (
    <AppBar position="static" color="default" elevation={1}>
      <Toolbar sx={{ flexDirection: "column", alignItems: "flex-start", py: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
          <img src="/pt-logo.svg" alt="PetreeBlitz" style={{ height: 80 }} />
          <Typography variant="h4" color="textSecondary" sx={{ ml: 1 }}>
            Archaeobotany AI Tree Identification
          </Typography>
        </Box>

        <Tabs
          value={current}
          textColor="primary"
          indicatorColor="primary"
          sx={{ width: "100%" }}
        >
          {navItems.map(item => (
            <Tab
              key={item.to}
              label={item.label}
              icon={item.icon}
              component={Link}
              to={item.to}
              value={item.to}
            />
          ))}
        </Tabs>
      </Toolbar>
    </AppBar>
  );
}
