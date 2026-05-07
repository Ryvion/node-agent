package modelpolicy

func StatusFromEnv() Status {
	return BuildStatus(FromEnv())
}

func BuildStatus(policy Policy) Status {
	return NormalizePolicy(policy)
}
