SELECT gufi, departure_runway_actual_time, departure_runway_actual, arrival_runway_actual_time,arrival_runway_actual FROM runways
 where (departure_aerodrome_iata_name = SUBSTRING(:airport_icao,2,3) and departure_runway_actual_time between :start_time and :end_time) or
	(departure_aerodrome_iata_name = SUBSTRING(:airport_icao,2,3) and arrival_runway_actual_time between :start_time and :end_time) and
	(points_on_runway = :surf_surv_avail)
