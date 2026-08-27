class EnergyEstimator:

    @staticmethod
    def estimate_joules(latency_ms: float, power_watts: float = 2.5) -> float:
        latency_seconds = latency_ms / 1000.0
        return latency_seconds * power_watts
