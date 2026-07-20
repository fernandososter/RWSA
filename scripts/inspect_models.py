from sleep_rswa.models.staging import SleepStagingNet
from sleep_rswa.models.rswa import RSWADetectionNet
from sleep_rswa.models.system import SleepStagingRSWASystem

staging=SleepStagingNet(); rswa=RSWADetectionNet(); system=SleepStagingRSWASystem(staging,rswa)
print(f"Staging: {staging.n_params():,} parâmetros")
print(f"RSWA:    {rswa.n_params():,} parâmetros")
print(f"Total:   {system.n_params():,} parâmetros")
print(system)
