function y = is_polarized(rac)
  % #takes rac and rho, two vectors which #describe concentration over space.
  % #Determine if polarized pattern exists...

  tol = 0.1;

  if abs(max(rac) - min(rac)) > tol
      y = 1;
  else
    y = 0;
  end
end
