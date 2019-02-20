function [R,RR,v] = GenerateDiag(ss,k)

r1 = rand(ss,1);
r2 = rand(ss,1);
v = rand(ss,1);
R=r1*r2';
RR = spdiags(R,k);
R = full(spdiags(RR,k,ss,ss));
[RRn,RRm]=size(RR);
for c = 1:RRm
  if k(c) < 0         
    for f = RRn : -1 : 1       
      if f + k(c) > 0     
        RR(f,c)=RR(f+k(c),c);
      else
        RR(f,c) = 0;
      end
    end
  end
  if k(c) > 0         
    for f = 1:RRn       
      if f + k(c) <= RRn    
        RR(f,c)=RR(f+k(c),c);
      else
        RR(f,c) = 0;
      end
    end
  end
end
end

