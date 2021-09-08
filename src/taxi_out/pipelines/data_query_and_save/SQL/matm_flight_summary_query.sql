with mfa as (
	select
         gufi,
	 carrier,
         acid,
         departure_stand_initial_time,
	 aircraft_type,
	 arrival_aerodrome_icao_name,
         departure_aerodrome_icao_name,
	 case when (arrival_aerodrome_icao_name = :airport_icao) then True
		  else False
	 end as isArrival,
	 case when (departure_aerodrome_icao_name = :airport_icao) then True
		  else False
	 end as isDeparture,
	 COALESCE(arrival_stand_actual,
		arrival_stand_user,
		arrival_stand_airline) as arrival_stand_actual,
	 arrival_stand_actual_time,
	 arrival_stand_airline_time,
	 COALESCE(arrival_spot_actual,
		arrival_spot_user,
		arrival_spot_airline) as arrival_spot_actual,
	 arrival_movement_area_actual_time,
	 COALESCE(arrival_runway_actual,
	    arrival_runway_user,
		arrival_runway_assigned,
		arrival_runway_airline) as arrival_runway_actual,
	 arrival_runway_actual_time,
	 COALESCE(departure_stand_actual,
		departure_stand_user,
		departure_stand_airline) as departure_stand_actual,
	 departure_stand_actual_time,
	 departure_stand_airline_time,
	 COALESCE(departure_spot_actual,
		departure_spot_user,
		departure_spot_airline) as departure_spot_actual,
	 departure_movement_area_actual_time,
	 COALESCE(departure_runway_actual,
	    departure_runway_user,
		departure_runway_assigned,
		departure_runway_airline) as departure_runway_actual,
	 departure_runway_actual_time
	from 
	 matm_flight_summary
	where 
	(departure_aerodrome_icao_name = :airport_icao and departure_runway_actual_time between :start_time and :end_time) or
	(arrival_aerodrome_icao_name = :airport_icao and 
	  (arrival_stand_actual_time between :start_time and :end_time or 
	   arrival_runway_actual_time between :start_time and :end_time or
	   arrival_movement_area_actual_time between :start_time and :end_time))
)
select
 extract(epoch from mfa.arrival_stand_actual_time - mfa.arrival_movement_area_actual_time) as actual_arrival_ramp_taxi_time,
 extract(epoch from mfa.arrival_movement_area_actual_time - mfa.arrival_runway_actual_time) as actual_arrival_ama_taxi_time,
 extract(epoch from mfa.departure_movement_area_actual_time -mfa.departure_stand_actual_time ) as actual_departure_ramp_taxi_time,
 extract(epoch from mfa.departure_runway_actual_time - mfa.departure_movement_area_actual_time) as actual_departure_ama_taxi_time,
 extract(epoch from mfa.departure_runway_actual_time - mfa.departure_stand_actual_time) as actual_departure_full_taxi_time,
 mfa.* 
from 
 mfa
