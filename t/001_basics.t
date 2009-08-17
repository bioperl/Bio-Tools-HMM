## Test framework for HMM XS stuff
## $Id$
use strict;
use warnings;

use Bio::Matrix::Scoring;
use Bio::Tools::HMM;

my $hmm = Bio::Tools::HMM->new('-symbols' => "123456", '-states' => "FL");

my ($seq1, $obs1);
$seq1 =  "315116246446644245311321631164152133625144543631656626566666";
$obs1 =  "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLL";
$seq1 .= "651166453132651245636664631636663162326455236266666625151631";
$obs1 .= "LLLLLLFFFFFFFFFFFFLLLLLLLLLLLLLLLLFFFLLLLLLLLLLLLLLFFFFFFFFF";
$seq1 .= "222555441666566563564324364131513465146353411126414626253356";
$obs1 .= "FFFFFFFFLLLLLLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
$seq1 .= "366163666466232534413661661163252562462255265252265435353336";
$obs1 .= "LLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
$seq1 .= "233121625364414432335163243633665562466662632666612355245242";
$obs1 .= "FFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLLLLLLLLLFFFFFFFFFFF";

my ($seq2, $obs2);
$seq2 =  "544552213525245666363632432522253566166546666666533666543261";
$obs2 =  "FFFFFFFFFFFFLLLLLLLLLLLFFFFFFFFFFFFLLLLLLLLLLLLLLLLLLLFFFFFF";
$seq2 .= "363546253252546524422555242223224344432423341365415551632161";
$obs2 .= "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
$seq2 .= "144212242323456563652263346116214136666156616666566421456123";
$obs2 .= "FFFFFFLLLFFFFFFFFFFFFFFFFFFFFFFFFLFLLLLLLLLLLLLLLLLFFFFFFFFF";
$seq2 .= "346313546514332164351242356166641344615135266642261112465663";
$obs2 .= "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

my @seqs = ($seq1, $seq2);

printf "Baum-Welch Training\n";
printf "===================\n";


$hmm->baum_welch_training(\@seqs);

printf "Initial Probability Array:\n";
my $init = $hmm->init_prob;
foreach my $s (@{$init}) {
   printf "%g\t", $s;
}
printf "\n";

printf "Transition Probability Matrix:\n";
my $matrix = $hmm->transition_prob;
foreach my $r ($matrix->row_names) {
   foreach my $c ($matrix->column_names) {
      printf "%g\t", $matrix->entry($r, $c);
   }
   printf "\n";
}

printf "Emission Probability Matrix:\n";
$matrix = $hmm->emission_prob;
foreach my $r ($matrix->row_names) {
   foreach my $c ($matrix->column_names) {
      printf "%g\t", $matrix->entry($r, $c);
   }
   printf "\n";
}

printf "\n";
printf "Log Probability of sequence 1: %g\n", $hmm->likelihood($seq1);
printf "Log Probability of sequence 2: %g\n", $hmm->likelihood($seq2);
printf "\n";
printf "Statistical Training\n";
printf "====================\n";

my @obs = ($obs1, $obs2);
$hmm->statistical_training(\@seqs, \@obs);

printf "Initial Probability Array:\n";
$init = $hmm->init_prob;
$hmm->init_prob($init);
foreach my $s (@{$init}) {
   printf "%g\t", $s;
}
printf "\n";

printf "Transition Probability Matrix:\n";
$matrix = $hmm->transition_prob;
$hmm->transition_prob($matrix);
foreach my $r ($matrix->row_names) {
   foreach my $c ($matrix->column_names) {
      printf "%g\t", $matrix->entry($r, $c);
   }
   printf "\n";
}

printf "Emission Probability Matrix:\n";
$matrix = $hmm->emission_prob;
$hmm->emission_prob($matrix);
foreach my $r ($matrix->row_names) {
   foreach my $c ($matrix->column_names) {
      printf "%g\t", $matrix->entry($r, $c);
   }
   printf "\n";
}

printf "Viterbi Algorithm:\n";
my $obs3 = $hmm->viterbi($seq1);
printf "%s\n", $obs3;
