#ifndef TRAIN_H_
#define TRAIN_H_

namespace igd {

	struct TrainArgs {
		TrainArgs()
				: db_ptr(nullptr)
				, root(keen::next_ident("train"))
				, icol("intercept") {
			calc_tab = keen::str("%1%_calculate", root);
			calc_agg = keen::str("%1%_aggregate", root);
		}
		sqlite3 *db_ptr;
		std::string tabname;
		std::string ycol;
		std::string icol;
		std::vector<std::string> xcols;

		// generated
		std::string root;
		std::string calc_tab;
		std::string calc_agg;
	};

	class AggregateVisitor {
	public:
		virtual ~AggregateVisitor() {}
		virtual void step(const std::vector<double> &record) = 0;
		virtual void stop() = 0;
		virtual bool done() = 0;
	};

	void train(const TrainArgs &args, AggregateVisitor *vis);

} // igd

#endif

