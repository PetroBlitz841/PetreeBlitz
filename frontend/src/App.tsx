import { Box } from "@mui/material";
import Header from "./components/layout/Header";
import Footer from "./components/layout/Footer";
import { Routes, Route } from "react-router-dom";

import IdentifyPage from "./pages/IdentifyPage";
import AlbumsPage from "./pages/AlbumsPage";
import AlbumDetailsPage from "./pages/AlbumDetailsPage";
import DashboardPage from "./pages/DashboardPage";

function App() {
  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        backgroundColor: "grey.50",
      }}
    >
      <Header />

      <Box sx={{ flex: 1 }}>
        <Routes>
          <Route path="/" element={<IdentifyPage />} />
          <Route path="/albums" element={<AlbumsPage />} />
          <Route path="/albums/:id" element={<AlbumDetailsPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </Box>

      <Footer />
    </Box>
  );
}

export default App;
