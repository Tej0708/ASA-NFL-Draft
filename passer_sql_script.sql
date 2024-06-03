set search_path to nfl;

select * from passer;

select "playerId" d, sum("passLength") as passYards, sum("passAtt") as passAttempts, sum("passTd")
as passTd, sum(p."passInt") as passInt
from passer p 
group by "playerId";

create table nfl.passing (
playerId varchar(8),
passYards int8,
passAttempts int8,
passTd int8,
passInt int8
)

insert into passing(
playerId, passYards, passAttempts, passTd, passInt
)
select "playerId" as playerId, sum("passLength") as passYards, sum("passAtt") as passAttempts, sum("passTd")
as passTd, sum(p."passInt") as passInt
from passer p 
group by "playerId";

select *
from passing p inner join combine c on c."playerId" = p.playerid;

select "playerId" as playerId, sum("recYards") as recYards, sum("rec") as rec, avg("recYac") as avg_Yac
from receiver
where "recPosition" = 'WR'
group by "playerId";


create table nfl.receiving (
playerId varchar(8),
recYards int8,
rec int8,
recYac 
)

insert into receiving(
playerId, recYards, rec, avg_Yac
)
select "playerId" as playerId, sum("recYards") as recYards, sum("rec") as rec, avg("recYac") as avg_Yac
from receiver
where "recPosition" = 'WR'
group by "playerId";




