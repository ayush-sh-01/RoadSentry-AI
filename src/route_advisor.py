"""
RoadSentry AI - Emergency Route & Intersection Assistance Module
Provides intelligent traffic signal preemption, green-wave corridor guidance,
and detour recommendations when emergency vehicles (Ambulance, Fire, Police) are identified.
"""

from typing import List, Dict, Any, Optional


class EmergencyRouteAdvisor:
    """
    Emergency Route Guidance & Intersection Clearance Advisor.
    Calculates priority green-light corridors, civilian traffic diversions,
    and route recommendations.
    """

    def __init__(
        self,
        priority_corridor: str = "North Approach -> Central Ring -> East Hospital Lane",
        green_wave_seconds: int = 35,
        detour_recommendation: str = "Civilian traffic: Divert to 80ft Outer Bypass",
        enabled: bool = True,
        **kwargs
    ):
        self.priority_corridor = priority_corridor
        self.green_wave_seconds = green_wave_seconds
        self.detour_recommendation = detour_recommendation
        self.enabled = enabled

    def get_assistance(
        self,
        emergencies: List[Dict[str, Any]],
        location_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes intersection signal priority and route clearance advisories.
        """
        loc = location_data or {
            "junction_name": "Central Ring Road Junction",
            "latitude": 12.9716,
            "longitude": 77.5946
        }

        lat = float(loc.get("latitude", 12.9716))
        lon = float(loc.get("longitude", 77.5946))

        if emergencies:
            primary_em = emergencies[0]
            vehicle_id = primary_em.get("track_id", "Unknown")
            subtype = primary_em.get("subtype", "ambulance").upper()

            # Priority Route Waypoints (Monitored Junction -> Hospital / Clear Corridor)
            corridor_waypoints = [
                {"lat": lat + 0.004, "lon": lon - 0.003, "label": "North Inflow Approach"},
                {"lat": lat, "lon": lon, "label": loc.get("junction_name", "Current Junction")},
                {"lat": lat - 0.003, "lon": lon + 0.004, "label": "East Emergency Lane (Hospital)"}
            ]

            return {
                "is_active": True,
                "vehicle_id": vehicle_id,
                "vehicle_type": subtype,
                "signal_phase": "GREEN_WAVE_PRIORITY",
                "phase_badge_color": "#00e676",
                "intersection_action": f"🚨 EMERGENCY GREEN WAVE: Lane clear priority enforced for {subtype} #{vehicle_id}",
                "corridor_name": self.priority_corridor,
                "green_duration": f"{self.green_wave_seconds} seconds extended",
                "cross_traffic_status": "Cross-traffic signals set to HOLD RED (Intersection Clearance Enforced)",
                "detour_advisory": self.detour_recommendation,
                "target_destination": "Metro District Hospital Corridor",
                "corridor_waypoints": corridor_waypoints
            }
        else:
            return {
                "is_active": False,
                "vehicle_id": None,
                "vehicle_type": None,
                "signal_phase": "NORMAL_CYCLIC",
                "phase_badge_color": "#94a3b8",
                "intersection_action": "🟢 NORMAL CYCLIC TIMING: Standard 45s signal rotation active",
                "corridor_name": "Standby (No Emergency Vehicles in Zone)",
                "green_duration": "45s standard phase",
                "cross_traffic_status": "Standard dynamic sequencing active",
                "detour_advisory": "Normal traffic flow. No diversions required.",
                "target_destination": "All corridors clear",
                "corridor_waypoints": []
            }
