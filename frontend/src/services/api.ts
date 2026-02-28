import axios from "axios";

// centralized axios instance for all API requests
// baseURL will automatically prefix "/api" so calls can use relative paths
const api = axios.create({
  baseURL: "/api",
});

export default api;