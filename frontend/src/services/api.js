/**
 * API client service for communicating with NETRA FastAPI backend.
 * Placeholder for future implementation.
 */

const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000';

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/`);
  return response.json();
}

export async function fetchNetworkStats() {
  const response = await fetch(`${API_BASE_URL}/network/stats`);
  return response.json();
}

export async function postPrediction(features) {
  const response = await fetch(`${API_BASE_URL}/predict/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  });
  return response.json();
}

export async function postForecast(history) {
  const response = await fetch(`${API_BASE_URL}/forecast/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ history, window_count: history.length }),
  });
  return response.json();
}
