using WaspNet

adexp = WaspNet.ADEXP()
v = adexp.v
println(v,adexp)
adexp = update(adexp, 1, 0, 0)
