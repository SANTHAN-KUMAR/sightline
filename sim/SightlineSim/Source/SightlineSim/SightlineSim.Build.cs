// Sightline simulation - primary game module.

using UnrealBuildTool;

public class SightlineSim : ModuleRules
{
	public SightlineSim(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		// AirSim headers use C++ exceptions on Win64 (same as the upstream Blocks project).
		bEnableExceptions = Target.Platform != UnrealTargetPlatform.Linux;
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" });
	}
}
