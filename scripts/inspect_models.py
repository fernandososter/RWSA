from sleep_rswa import SleepStagingNet, RSWADetectionNet, SleepStagingRSWASystem

staging=SleepStagingNet(); rswa=RSWADetectionNet(); system=SleepStagingRSWASystem(staging,rswa)
print(f"Staging: {staging.n_params():,} parâmetros")
print(f"RSWA:    {rswa.n_params():,} parâmetros")
print(f"Total:   {system.n_params():,} parâmetros")
print(system)
