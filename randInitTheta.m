%#% Project Opinion Mining - Author Arun Karthikeyan %#%
function initTheta = randInitTheta(in, out)

einit = (6^(0.5))/((in+out)^(0.5));
initTheta = (rand(in,out)*2*einit)-einit;

end
