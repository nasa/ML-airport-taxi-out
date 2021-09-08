select
        mall.gufi as gufi,
        AVG((mall.position_speed >= :threshold_knots)::int::float4) as fraction_speed_gte_threshold
from matm_position_all mall
inner join matm_flight_summary msum on mall.gufi = msum.gufi
where 
      msum.departure_aerodrome_icao_name = :airport_icao
      and (msum.departure_runway_actual_time between :start_time and :end_time)
      and mall.system_id = 'ASDEX'
      and mall.timestamp > (:start_time - '1 hour'::interval)
      and mall.timestamp < (:end_time + '1 hour'::interval)
      and mall.timestamp < msum.departure_runway_actual_time
      and ((mall.timestamp > msum.departure_movement_area_actual_time and :surf_surv_avail = 'True')
      or (:surf_surv_avail = 'False'))
group by mall.gufi
