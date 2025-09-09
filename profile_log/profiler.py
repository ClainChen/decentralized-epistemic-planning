import pstats

p = pstats.Stats('profile_log/result2.prof')

print(p.total_tt)