from sleep_rswa import SleepStagingNet, RSWADetectionNet, SleepStagingRSWASystem

staging=SleepStagingNet(); rswa=RSWADetectionNet(stage_conditioning=False); system=SleepStagingRSWASystem()
print(f"Staging:          {staging.n_params():,} parâmetros")
print(f"RSWA standalone:  {rswa.n_params():,} parâmetros")
print(f"Joint system:     {system.n_params():,} parâmetros")
print(system)
